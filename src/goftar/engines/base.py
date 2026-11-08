"""The contract every ASR backend implements."""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path

from goftar.core.config import Config
from goftar.core.types import Transcript


class ASREngine(ABC):
    """An ASR backend. Receives engine-ready audio, returns raw (un-normalised) text."""

    name: str = "base"

    def __init__(self, config: Config) -> None:
        self.config = config

    def load(self) -> None:  # noqa: B027 - optional hook, intentionally not abstract
        """Load model weights eagerly (lets benchmarks time loading separately)."""

    @abstractmethod
    def transcribe(self, audio_path: str | Path) -> Transcript:
        """Transcribe one audio file."""
