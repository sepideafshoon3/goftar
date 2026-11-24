"""Live microphone transcription: chunk handling, tentative text and commits."""

import numpy as np
import pytest

from goftar.core.pipeline import Pipeline
from goftar.errors import EngineError
from goftar.live import SAMPLE_RATE, LiveTranscriber, quietest_cut, to_mono_16k


def tone(seconds, level=0.3, sr=SAMPLE_RATE):
    t = np.arange(int(seconds * sr)) / sr
    return (level * np.sin(2 * np.pi * 220 * t)).astype(np.float32)


def silence(seconds, sr=SAMPLE_RATE):
    return np.zeros(int(seconds * sr), np.float32)


class Recorder:
    """A decode function that records each call and names the text after it."""

    def __init__(self):
        self.calls = []

    def __call__(self, samples, fast):
        self.calls.append((round(len(samples) / SAMPLE_RATE, 1), fast))
        return f"{'fast' if fast else 'full'}{len(self.calls)}"


def feed_in_chunks(live, audio, sr=SAMPLE_RATE, chunk=0.5):
    step = int(chunk * sr)
    for i in range(0, len(audio), step):
        live.feed(sr, audio[i : i + step])


def test_to_mono_16k_handles_browser_formats():
    stereo_int16 = (np.ones((48000, 2)) * 16384).astype(np.int16)
    out = to_mono_16k(48000, stereo_int16)
    assert out.dtype == np.float32 and len(out) == 16000
    assert out[0] == pytest.approx(0.5, abs=1e-3)
    assert len(to_mono_16k(44100, np.zeros(44100, np.float32))) == 16000
    assert len(to_mono_16k(16000, np.zeros(10, np.float32))) == 10


def test_quietest_cut_lands_in_the_pause():
    x = np.concatenate([tone(2), silence(0.5), tone(2)])
    cut = quietest_cut(x, 0)
    assert 2.0 * SAMPLE_RATE <= cut <= 2.5 * SAMPLE_RATE


def test_silence_is_never_decoded():
    rec = Recorder()
    live = LiveTranscriber(rec)
    feed_in_chunks(live, silence(10))
    assert rec.calls == [] and live.finish() == ""


def test_speech_shows_tentative_text_then_commits_on_a_pause():
    rec = Recorder()
    live = LiveTranscriber(rec)
    feed_in_chunks(live, tone(3))
    assert live.tentative.startswith("fast") and live.committed == []
    assert all(fast for _, fast in rec.calls)
    feed_in_chunks(live, silence(1))
    assert live.committed and live.committed[-1].startswith("full")
    assert live.tentative == ""


def test_long_speech_without_pause_commits_before_whisper_limit():
    rec = Recorder()
    live = LiveTranscriber(rec)
    feed_in_chunks(live, tone(25))
    full = [sec for sec, fast in rec.calls if not fast]
    assert full and max(full) <= 12.5
    assert max(sec for sec, _ in rec.calls) < 30


def test_finish_commits_what_is_pending():
    rec = Recorder()
    live = LiveTranscriber(rec)
    feed_in_chunks(live, tone(1.5))
    text = live.finish()
    assert text.endswith(f"full{len(rec.calls)}")
    assert rec.calls[-1] == (1.5, False)


def test_pipeline_live_needs_an_engine_that_supports_it(fake_config):
    with pytest.raises(EngineError, match="cannot transcribe live audio"):
        Pipeline(fake_config).transcribe_samples(silence(1))
