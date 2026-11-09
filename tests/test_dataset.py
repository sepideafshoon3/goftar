import shutil

import pytest

from goftar.bench import discover_samples
from goftar.core.pipeline import Pipeline
from goftar.dataset import (
    DRAFT,
    FAILED,
    HAS_DRAFT,
    HAS_REFERENCE,
    MISSING,
    READY,
    WRITTEN,
    accept_draft,
    draft_path,
    list_entries,
    prepare_drafts,
    reference_path,
)
from tests.conftest import FakeEngine


@pytest.fixture
def folder(tmp_path, wav_file):
    d = tmp_path / "dataset"
    d.mkdir()
    shutil.copy(wav_file, d / "a.wav")
    shutil.copy(wav_file, d / "b.wav")
    (d / "notes.md").write_text("not audio", encoding="utf-8")
    return d


def test_paths():
    from pathlib import Path

    assert reference_path(Path("x/v.ogg")).name == "v.txt"
    assert draft_path(Path("x/v.ogg")).name == "v.draft.txt"
    assert draft_path(Path("x/a.b.ogg")).name == "a.b.draft.txt"


def test_list_entries_states(folder):
    reference_path(folder / "a.wav").write_text("ref", encoding="utf-8")
    draft_path(folder / "b.wav").write_text("draft", encoding="utf-8")
    assert [(e.audio.name, e.state) for e in list_entries(folder)] == [
        ("a.wav", READY),
        ("b.wav", DRAFT),
    ]
    draft_path(folder / "b.wav").unlink()
    assert list_entries(folder)[1].state == MISSING


def test_prepare_writes_drafts_one_segment_per_line(folder, fake_config):
    outcomes = prepare_drafts(Pipeline(fake_config), folder)
    assert [o.status for o in outcomes] == [WRITTEN, WRITTEN]
    lines = draft_path(folder / "a.wav").read_text(encoding="utf-8").splitlines()
    assert lines == ["سلام، حال شما چطوره؟", "من ۳ تا کتاب‌ها می‌خرم"]


def test_drafts_are_never_used_as_references(folder, fake_config):
    prepare_drafts(Pipeline(fake_config), folder)
    assert discover_samples(folder) == []


def test_prepare_skips_references_and_existing_drafts(folder, fake_config):
    reference_path(folder / "a.wav").write_text("ref", encoding="utf-8")
    draft_path(folder / "b.wav").write_text("my edited draft", encoding="utf-8")
    outcomes = prepare_drafts(Pipeline(fake_config), folder)
    assert [o.status for o in outcomes] == [HAS_REFERENCE, HAS_DRAFT]
    assert draft_path(folder / "b.wav").read_text(encoding="utf-8") == "my edited draft"
    assert FakeEngine.calls == 0


def test_prepare_overwrite_replaces_drafts_but_never_references(folder, fake_config):
    reference_path(folder / "a.wav").write_text("ref", encoding="utf-8")
    draft_path(folder / "b.wav").write_text("old", encoding="utf-8")
    outcomes = prepare_drafts(Pipeline(fake_config), folder, overwrite=True)
    assert [o.status for o in outcomes] == [HAS_REFERENCE, WRITTEN]
    assert reference_path(folder / "a.wav").read_text(encoding="utf-8") == "ref"
    assert "سلام" in draft_path(folder / "b.wav").read_text(encoding="utf-8")


def test_prepare_reports_failures_and_continues(folder, fake_config):
    (folder / "a.wav").write_bytes(b"garbage")
    # without ffmpeg the garbage is handed to the fake engine, which accepts anything
    from goftar.audio.ingest import ffmpeg_available

    outcomes = prepare_drafts(Pipeline(fake_config), folder)
    assert outcomes[1].status == WRITTEN
    if ffmpeg_available():
        assert outcomes[0].status == FAILED
        assert "could not decode" in outcomes[0].detail


def test_accept_promotes_draft(folder):
    draft_path(folder / "a.wav").write_text("corrected", encoding="utf-8")
    ref = accept_draft(folder / "a.wav")
    assert ref.read_text(encoding="utf-8") == "corrected"
    assert not draft_path(folder / "a.wav").exists()
    assert [s.audio.name for s in discover_samples(folder)] == ["a.wav"]


def test_accept_errors(folder):
    with pytest.raises(FileNotFoundError):
        accept_draft(folder / "a.wav")
    draft_path(folder / "a.wav").write_text("new", encoding="utf-8")
    reference_path(folder / "a.wav").write_text("old", encoding="utf-8")
    with pytest.raises(FileExistsError):
        accept_draft(folder / "a.wav")
    accept_draft(folder / "a.wav", force=True)
    assert reference_path(folder / "a.wav").read_text(encoding="utf-8") == "new"
