"""Live transcription of microphone audio that arrives in small chunks.

Audio is kept in a pending window. Every second or so the window is decoded quickly and shown
as tentative text. When the speaker pauses (or the window gets long), the window is decoded
once more with the full settings and its text is committed; only the rest stays pending.
Whisper reads at most 30 seconds at a time, so the window is kept well under that.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass, field

import numpy as np

SAMPLE_RATE = 16000
MIN_SECONDS = 1.0  # do not decode less audio than this
STEP_SECONDS = 1.0  # decode the tentative text again after this much new audio
PAUSE_SECONDS = 0.6  # this much quiet at the end of the window counts as a pause
MIN_COMMIT_SECONDS = 2.0  # a pause commits the window only when it is at least this long
MAX_SECONDS = 12.0  # commit even without a pause once the window is this long
SILENCE_RMS = 0.006  # below this level a stretch of audio is treated as silence
_FRAME = int(0.1 * SAMPLE_RATE)

Decode = Callable[[np.ndarray, bool], str]
"""decode(samples, fast) -> text, for 16 kHz mono float32 samples."""


def to_mono_16k(sample_rate: int, data: np.ndarray) -> np.ndarray:
    """A chunk from the browser (int16 or float, mono or stereo, any rate) as 16 kHz float32."""
    x = np.asarray(data)
    if np.issubdtype(x.dtype, np.integer):  # scale before mixing: mean() makes it float
        x = x.astype(np.float32) / float(np.iinfo(x.dtype).max)
    x = x.astype(np.float32, copy=False)
    if x.ndim == 2:  # (samples, channels)
        x = x.mean(axis=1)
    if sample_rate == SAMPLE_RATE or len(x) == 0:
        return x
    ratio = sample_rate / SAMPLE_RATE
    if ratio == int(ratio):  # 48 kHz -> 16 kHz: average each group (a crude low-pass)
        n = int(ratio)
        return x[: len(x) // n * n].reshape(-1, n).mean(axis=1)
    out_len = int(len(x) / ratio)
    return np.interp(np.arange(out_len) * ratio, np.arange(len(x)), x).astype(np.float32)


def rms(x: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(x)))) if len(x) else 0.0


def quietest_cut(x: np.ndarray, after: int) -> int:
    """Index of the middle of the quietest 0.1 s frame at or after `after` samples."""
    tail = x[after:]
    frames = tail[: len(tail) // _FRAME * _FRAME].reshape(-1, _FRAME)
    if len(frames) == 0:
        return len(x)
    return after + int(np.argmin(np.square(frames).mean(axis=1))) * _FRAME + _FRAME // 2


@dataclass
class LiveTranscriber:
    """Holds one live session. Not thread-safe: one per microphone stream."""

    decode: Decode
    committed: list[str] = field(default_factory=list)
    tentative: str = ""
    _pending: np.ndarray = field(default_factory=lambda: np.zeros(0, np.float32))
    _decoded_len: int = 0  # length of _pending when it was last decoded
    _last_decode_s: float = 0.0  # how long the last tentative decode took
    heard: float = 0.0  # seconds of audio received

    @property
    def text(self) -> str:
        return " ".join(t for t in self.committed if t)

    def feed(self, sample_rate: int, data: np.ndarray) -> None:
        chunk = to_mono_16k(sample_rate, data)
        self.heard += len(chunk) / SAMPLE_RATE
        self._pending = np.concatenate([self._pending, chunk])
        n = len(self._pending)
        if n < MIN_SECONDS * SAMPLE_RATE:
            return

        if rms(self._pending) < SILENCE_RMS:  # nothing said yet: keep only the last second
            self._pending = self._pending[-int(MIN_SECONDS * SAMPLE_RATE) :]
            self._decoded_len = 0
            self.tentative = ""
            return

        pause = int(PAUSE_SECONDS * SAMPLE_RATE)
        paused = n >= MIN_COMMIT_SECONDS * SAMPLE_RATE and rms(self._pending[-pause:]) < max(
            SILENCE_RMS, 0.15 * rms(self._pending)
        )
        if paused:
            self._commit(n)
        elif n >= MAX_SECONDS * SAMPLE_RATE:
            # cut in the quietest place of the last third, so a word is not split in two
            self._commit(quietest_cut(self._pending, n * 2 // 3))
        else:
            # if decoding falls behind, decode less often instead of piling up work
            step = max(STEP_SECONDS, 2 * self._last_decode_s) * SAMPLE_RATE
            if n - self._decoded_len >= step:
                started = time.perf_counter()
                self.tentative = self.decode(self._pending, True)
                self._last_decode_s = time.perf_counter() - started
                self._decoded_len = n

    def finish(self) -> str:
        """Commit whatever is still pending and return the full text."""
        if len(self._pending) >= 0.3 * SAMPLE_RATE and rms(self._pending) >= SILENCE_RMS:
            self._commit(len(self._pending))
        self._pending = np.zeros(0, np.float32)
        self.tentative = ""
        return self.text

    def _commit(self, cut: int) -> None:
        text = self.decode(self._pending[:cut], False)
        if text:
            self.committed.append(text)
        self._pending = self._pending[cut:]
        self._decoded_len = 0
        self.tentative = ""
