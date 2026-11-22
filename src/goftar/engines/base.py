"""The contract every ASR backend implements."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable
from pathlib import Path
from typing import Any

from goftar.core.config import Config
from goftar.core.types import Segment, Transcript

ProgressFn = Callable[[float], None]
"""Called with the fraction of the audio done so far (0.0 to 1.0)."""

SegmentFn = Callable[[Segment], None]
"""Called with each raw segment as soon as the engine has it (for live display)."""


class ASREngine(ABC):
    """An ASR backend. Receives engine-ready audio, returns raw (un-normalised) text."""

    name: str = "base"

    def __init__(self, config: Config) -> None:
        self.config = config

    def load(self) -> None:  # noqa: B027 - optional hook, intentionally not abstract
        """Load model weights eagerly (lets benchmarks time loading separately)."""

    @abstractmethod
    def transcribe(self, audio_path: str | Path, progress: ProgressFn | None = None) -> Transcript:
        """Transcribe one audio file. Engines that can report progress call `progress`.

        Engines that can stream also accept `on_segment` (a :data:`SegmentFn`) as a keyword.
        """

    def transcribe_samples(self, samples: Any, *, fast: bool = False) -> list[Segment]:
        """Transcribe 16 kHz mono float32 samples held in memory (live microphone audio).

        `fast` trades a little accuracy for speed (used for text that is shown, then redone).
        """
        raise NotImplementedError(f"the {self.name!r} engine cannot transcribe live audio")
