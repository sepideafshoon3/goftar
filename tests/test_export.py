import json

import pytest

from goftar.core.types import Segment, Transcript
from goftar.export.formats import (
    FORMATS,
    MAX_LINE_CHARS,
    format_timestamp,
    render,
    subtitle_cues,
    to_srt,
    to_vtt,
    wrap_lines,
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


LONG = (
    "مثلا میتونه سلیغه روی لباس پوشیدن آدم‌ها باشه، چجور آدم‌ها دوست دارن خونه شون رو "
    "دکوریت کنن، چه لوازم خونه‌ای دوست دارن و چه جوری دوست دارن مسافرت برن و چه فیلمی ببینن"
)


def test_short_cue_is_unchanged():
    assert subtitle_cues(Segment(1.0, 2.0, "سلام")) == [(1.0, 2.0, "سلام")]


def test_medium_cue_is_wrapped_in_two_balanced_lines():
    text = "یه کارایی هست که آدم‌ها دوست دارن یکی دیگه برشون انجام بده"
    [(_, _, wrapped)] = subtitle_cues(Segment(0.0, 4.0, text))
    lines = wrapped.split("\n")
    assert len(lines) == 2 and " ".join(lines) == text
    assert all(len(line) <= MAX_LINE_CHARS for line in lines)


def test_wrap_prefers_a_break_after_punctuation():
    assert wrap_lines("سلام علیکم دوستان، امروز درباره‌ی پروژه حرف می‌زنیم") == (
        "سلام علیکم دوستان،\nامروز درباره‌ی پروژه حرف می‌زنیم"
    )


def test_long_segment_becomes_several_timed_cues():
    cues = subtitle_cues(Segment(10.0, 22.0, LONG))
    assert len(cues) >= 2
    assert cues[0][0] == 10.0 and cues[-1][1] == 22.0
    assert all(a < b for a, b, _ in cues)
    assert all(prev[1] == pytest.approx(nxt[0]) for prev, nxt in zip(cues, cues[1:], strict=False))
    assert " ".join(" ".join(c[2].split("\n")) for c in cues) == LONG  # no word lost
    for _, _, text in cues:
        assert len(text.split("\n")) <= 2
        assert all(len(line) <= MAX_LINE_CHARS + 10 for line in text.split("\n"))


def test_srt_numbers_split_cues_in_order():
    t = Transcript(segments=[Segment(0.0, 12.0, LONG), Segment(12.0, 13.0, "تموم")])
    srt = to_srt(t)
    numbers = [b.split("\n")[0] for b in srt.strip().split("\n\n")]
    assert numbers == [str(i) for i in range(1, len(numbers) + 1)]
    assert srt.strip().endswith("تموم")
