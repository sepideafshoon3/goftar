"""Data structures shared by every stage of the pipeline."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass(frozen=True)
class Segment:
    """A time-aligned piece of transcribed speech."""

    start: float
    end: float
    text: str
    confidence: float | None = None
    speaker: str | None = None


@dataclass
class Transcript:
    """Result of transcribing one audio file."""

    segments: list[Segment] = field(default_factory=list)
    language: str = "fa"
    duration: float | None = None
    engine: str = ""
    model: str = ""
    processing_time: float | None = None

    @property
    def text(self) -> str:
        return " ".join(s.text for s in self.segments if s.text).strip()

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["text"] = self.text
        return data

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Transcript:
        return cls(
            segments=[Segment(**s) for s in data.get("segments", [])],
            language=data.get("language", "fa"),
            duration=data.get("duration"),
            engine=data.get("engine", ""),
            model=data.get("model", ""),
            processing_time=data.get("processing_time"),
        )
