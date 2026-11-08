from __future__ import annotations

import wave
from pathlib import Path

import pytest

from goftar.core.config import Config
from goftar.core.types import Segment, Transcript
from goftar.engines import ASREngine, register_engine


class FakeEngine(ASREngine):
    """Deterministic engine: returns canned segments with typical raw-Whisper flaws."""

    name = "fake"
    calls = 0

    def transcribe(self, audio_path):
        type(self).calls += 1
        return Transcript(
            segments=[
                Segment(0.0, 1.5, "سلام, حال شما چطوره ?"),
                Segment(1.5, 3.0, "من 3 تا كتاب ها می خرم"),
                Segment(3.0, 3.5, "   "),
            ],
            language="fa",
            duration=3.5,
            engine=self.name,
            model="fake-model",
        )


@pytest.fixture(autouse=True)
def _register_fake_engine():
    FakeEngine.calls = 0
    register_engine("fake", FakeEngine)


@pytest.fixture
def fake_config(tmp_path: Path) -> Config:
    cfg = Config(engine="fake")
    cfg.cache.path = str(tmp_path / "cache.sqlite")
    return cfg


@pytest.fixture
def wav_file(tmp_path: Path) -> Path:
    """One second of 16 kHz mono silence: a real, decodable WAV."""
    path = tmp_path / "voice.wav"
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(16000)
        w.writeframes(b"\x00\x00" * 16000)
    return path
