import json

import pytest

from goftar.core.types import Segment, Transcript
from goftar.export.formats import (
    FORMATS,
    format_timestamp,
    render,
    to_srt,
    to_vtt,
    write_transcript,
)


@pytest.fixture
def transcript():
    return Transcript(
        segments=[Segment(0.0, 1.5, "سلام"), Segment(61.25, 3723.5, "خداحافظ"), Segment(4, 5, "")],
        duration=3723.5,
        engine="fake",
        model="m",
    )


@pytest.mark.parametrize(
    ("seconds", "sep", "expected"),
    [(0, ",", "00:00:00,000"), (1.5, ".", "00:00:01.500"), (3723.5, ",", "01:02:03,500")],
)
def test_format_timestamp(seconds, sep, expected):
    assert format_timestamp(seconds, sep) == expected


def test_srt(transcript):
    assert to_srt(transcript) == (
        "1\n00:00:00,000 --> 00:00:01,500\nسلام\n\n2\n00:01:01,250 --> 01:02:03,500\nخداحافظ\n"
    )


def test_vtt(transcript):
    out = to_vtt(transcript)
    assert out.startswith("WEBVTT\n")
    assert "00:00:00.000 --> 00:00:01.500\nسلام" in out


def test_txt_skips_empty_segments(transcript):
    assert render(transcript, "txt") == "سلام\nخداحافظ\n"


def test_json_roundtrip_keeps_unicode(transcript):
    raw = render(transcript, "json")
    assert "سلام" in raw  # not \u-escaped
    data = json.loads(raw)
    assert data["text"] == "سلام خداحافظ"
    assert Transcript.from_dict(data).segments[1].start == 61.25


def test_unknown_format(transcript):
    with pytest.raises(ValueError, match="unknown format"):
        render(transcript, "docx")


def test_write_transcript(tmp_path, transcript):
    paths = write_transcript(transcript, tmp_path / "out", "voice", FORMATS)
    assert sorted(p.name for p in paths) == ["voice.json", "voice.srt", "voice.txt", "voice.vtt"]
    assert (tmp_path / "out" / "voice.txt").read_text(encoding="utf-8").startswith("سلام")
