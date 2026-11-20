"""Long audio is transcribed in ~10-minute pieces cut at quiet moments (bounded memory)."""

from types import SimpleNamespace

import numpy as np
import pytest

from goftar.core.config import Config
from goftar.engines import faster_whisper_engine as fw

SR = fw.SAMPLE_RATE


def noise(seconds: float) -> np.ndarray:
    return np.random.default_rng(0).uniform(-0.5, 0.5, int(seconds * SR)).astype(np.float32)


def test_short_audio_is_one_piece():
    audio = noise(60)
    assert fw.chunk_bounds(audio, chunk_seconds=600) == [0, len(audio)]


def test_cut_lands_in_the_pause_near_the_boundary():
    audio = noise(20)
    pause_at = 11.3  # a 0.5 s pause, 1.3 s after the 10 s target
    audio[int(pause_at * SR) : int((pause_at + 0.5) * SR)] = 0
    bounds = fw.chunk_bounds(audio, chunk_seconds=10, search_seconds=3)
    assert len(bounds) == 3 and bounds[0] == 0 and bounds[-1] == len(audio)
    assert pause_at <= bounds[1] / SR <= pause_at + 0.5


def test_every_piece_is_near_the_chunk_size():
    audio = noise(95)
    bounds = fw.chunk_bounds(audio, chunk_seconds=20, search_seconds=4)
    sizes = [(b - a) / SR for a, b in zip(bounds, bounds[1:], strict=False)]
    assert all(16 <= s <= 24 for s in sizes[:-1])
    assert sizes[-1] <= 24 + 4


class StubModel:
    """Says one segment per piece, at 1-2 s into the piece; records what it was given."""

    def __init__(self):
        self.calls = []

    def transcribe(self, audio, **kwargs):
        self.calls.append((len(audio), kwargs["language"]))
        seg = SimpleNamespace(
            start=1.0, end=2.0, text=f" piece {len(self.calls)} ", avg_logprob=0.0
        )
        return iter([seg]), SimpleNamespace(language="fa", duration=len(audio) / SR)


def test_segments_are_shifted_to_the_original_timeline(monkeypatch):
    faster_whisper = pytest.importorskip("faster_whisper")
    audio = noise(50)
    monkeypatch.setattr(faster_whisper, "decode_audio", lambda *a, **k: audio)
    monkeypatch.setattr(fw, "CHUNK_SECONDS", 20)
    monkeypatch.setattr(fw, "SEARCH_SECONDS", 2)
    engine = fw.FasterWhisperEngine(Config(language="auto"))
    engine._model = StubModel()

    result = engine.transcribe("unused.wav")

    bounds = fw.chunk_bounds(audio, 20, 2)
    assert len(result.segments) == len(bounds) - 1 == 3
    for seg, start in zip(result.segments, bounds, strict=False):
        assert seg.start == pytest.approx(start / SR + 1.0)
    assert [s.text for s in result.segments] == ["piece 1", "piece 2", "piece 3"]
    assert result.duration == pytest.approx(50)
    # language is detected on the first piece, then kept
    assert [lang for _, lang in engine._model.calls] == [None, "fa", "fa"]


def test_progress_rises_to_one_across_pieces(monkeypatch):
    faster_whisper = pytest.importorskip("faster_whisper")
    audio = noise(50)
    monkeypatch.setattr(faster_whisper, "decode_audio", lambda *a, **k: audio)
    monkeypatch.setattr(fw, "CHUNK_SECONDS", 20)
    monkeypatch.setattr(fw, "SEARCH_SECONDS", 2)
    engine = fw.FasterWhisperEngine(Config())
    engine._model = StubModel()
    seen: list[float] = []

    engine.transcribe("unused.wav", progress=seen.append)

    assert seen == sorted(seen)
    assert 0 < seen[0] < 1
    assert seen[-1] == pytest.approx(1.0)
