"""The contract every ASR backend implements."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable
from pathlib import Path

from goftar.core.config import Config
from goftar.core.types import Transcript

ProgressFn = Callable[[float], None]
"""Called with the fraction of the audio done so far (0.0 to 1.0)."""


class ASREngine(ABC):
    """An ASR backend. Receives engine-ready audio, returns raw (un-normalised) text."""

    name: str = "base"

    def __init__(self, config: Config) -> None:
        self.config = config

    def load(self) -> None:  # noqa: B027 - optional hook, intentionally not abstract
        """Load model weights eagerly (lets benchmarks time loading separately)."""

    @abstractmethod
    def transcribe(self, audio_path: str | Path, progress: ProgressFn | None = None) -> Transcript:
        """Transcribe one audio file. Engines that can report progress call `progress`."""
