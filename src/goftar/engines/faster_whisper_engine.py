"""faster-whisper (CTranslate2) backend."""

from __future__ import annotations

import logging
import math
import os
from pathlib import Path
from typing import Any

from goftar.core.config import AUTO_COMPUTE_TYPE, Config
from goftar.core.types import Segment, Transcript
from goftar.engines.base import ASREngine, ProgressFn, SegmentFn
from goftar.errors import EngineError

log = logging.getLogger(__name__)

SAMPLE_RATE = 16000
# faster-whisper computes features for the whole input at once, so peak RAM grows with length
# (~45 MB per minute: 5 GB for a 74-minute file). Long audio is fed in pieces of about this
# size, cut at the quietest moment near each boundary; peak RAM then stays flat.
CHUNK_SECONDS = 600
SEARCH_SECONDS = 20  # look this far either side of a boundary for a quiet place to cut
_FRAME_SECONDS = 0.1


def chunk_bounds(
    audio: Any, chunk_seconds: float | None = None, search_seconds: float | None = None
) -> list[int]:
    """Sample offsets [0, cut1, ..., len] that split `audio` into pieces of ~chunk_seconds.

    Each cut is placed at the quietest 0.1 s frame within +-search_seconds of the target, so
    it falls in a pause between words rather than inside one.
    """
    import numpy as np

    chunk_seconds = CHUNK_SECONDS if chunk_seconds is None else chunk_seconds
    search_seconds = SEARCH_SECONDS if search_seconds is None else search_seconds
    total = len(audio)
    chunk, search = int(chunk_seconds * SAMPLE_RATE), int(search_seconds * SAMPLE_RATE)
    frame = int(_FRAME_SECONDS * SAMPLE_RATE)
    bounds = [0]
    while total - bounds[-1] > chunk + search:
        lo, hi = bounds[-1] + chunk - search, bounds[-1] + chunk + search
        window = np.asarray(audio[lo:hi], dtype=np.float32)
        frames = window[: len(window) // frame * frame].reshape(-1, frame)
        quietest = int(np.argmin((frames**2).mean(axis=1)))
        bounds.append(lo + quietest * frame + frame // 2)
    bounds.append(total)
    return bounds


def segment_confidence(segment: Any) -> float | None:
    """Mean word probability of a faster-whisper segment, else exp(avg_logprob)."""
    words = getattr(segment, "words", None)
    if words:
        return sum(w.probability for w in words) / len(words)
    if segment.avg_logprob is not None:
        return math.exp(segment.avg_logprob)
    return None


def local_model_dir(name: str, models_dir: str | Path) -> Path:
    """Where a model called `name` lives under models_dir (repo ids use '--' for '/')."""
    return Path(models_dir).expanduser() / name.replace("/", "--")


def cuda_available() -> bool:
    try:
        import ctranslate2
    except ImportError:
        return False
    try:
        return ctranslate2.get_cuda_device_count() > 0
    except Exception:  # broken CUDA install: behave as if there is no GPU
        log.debug("CUDA probe failed", exc_info=True)
        return False


def resolve_device(device: str, compute_type: str) -> tuple[str, str]:
    """Turn "auto" into a concrete device and compute type (ADR 3).

    auto device -> cuda if a GPU is visible, else cpu.
    auto compute -> int8_float16 on cuda (fits large-v3-turbo in 4 GB VRAM), int8 on cpu.
    Explicit values are passed through unchanged.
    """
    if device == "auto":
        device = "cuda" if cuda_available() else "cpu"
    if compute_type == "auto":
        compute_type = AUTO_COMPUTE_TYPE.get(device, "auto")
    return device, compute_type


def download_model(name: str, models_dir: str | Path) -> Path:
    """Download a model once (needs network) into models_dir for fully offline use later."""
    try:
        from faster_whisper import download_model as fw_download
    except ImportError as exc:
        raise EngineError("faster-whisper is not installed") from exc
    target = local_model_dir(name, models_dir)
    target.mkdir(parents=True, exist_ok=True)
    fw_download(name, output_dir=str(target))
    return target


class FasterWhisperEngine(ASREngine):
    name = "faster_whisper"

    def __init__(self, config: Config) -> None:
        super().__init__(config)
        self._model: Any = None

    def resolve_model(self) -> str:
        cfg = self.config
        direct = Path(cfg.model).expanduser()
        if direct.is_dir():
            return str(direct)
        local = local_model_dir(cfg.model, cfg.models_dir)
        if local.is_dir():
            return str(local)
        if os.environ.get("HF_HUB_OFFLINE") == "1":
            raise EngineError(
                f"model {cfg.model!r} not found in {local.parent} and offline mode is on. "
                f"Run `goftar models download {cfg.model}` once while online."
            )
        log.warning("model %r not found locally; it will be downloaded from HuggingFace", cfg.model)
        return cfg.model

    def load(self) -> None:
        if self._model is not None:
            return
        try:
            from faster_whisper import WhisperModel
        except ImportError as exc:
            raise EngineError(
                "faster-whisper is not installed (pip install faster-whisper)"
            ) from exc
        cfg = self.config
        source = self.resolve_model()
        device, compute_type = resolve_device(cfg.device, cfg.compute_type)
        log.info("loading %s on %s (%s)", source, device, compute_type)
        self._model = WhisperModel(
            source,
            device=device,
            compute_type=compute_type,
            cpu_threads=cfg.cpu_threads,
        )

    def _decode_kwargs(self) -> dict[str, Any]:
        cfg = self.config
        kwargs: dict[str, Any] = {
            "language": None if cfg.language == "auto" else cfg.language,
            "beam_size": cfg.beam_size,
            "vad_filter": cfg.vad.enabled,
            "condition_on_previous_text": cfg.condition_on_previous_text,
            "initial_prompt": cfg.initial_prompt or None,
            # Per-word probabilities give each segment its own confidence (avg_logprob is
            # one value per 30-second window). Measured: same text, no extra time.
            "word_timestamps": True,
        }
        if cfg.vad.enabled:
            kwargs["vad_parameters"] = {"min_silence_duration_ms": cfg.vad.min_silence_ms}
        return kwargs

    def transcribe_samples(self, samples: Any, *, fast: bool = False) -> list[Segment]:
        self.load()
        kwargs = self._decode_kwargs()
        if fast:  # shown for a moment, then redone with the full settings
            kwargs.update(beam_size=1, word_timestamps=False)
        segment_iter, _ = self._model.transcribe(samples, **kwargs)
        return [
            Segment(float(s.start), float(s.end), s.text.strip(), segment_confidence(s))
            for s in segment_iter
        ]

    def transcribe(
        self,
        audio_path: str | Path,
        progress: ProgressFn | None = None,
        on_segment: SegmentFn | None = None,
    ) -> Transcript:
        self.load()
        cfg = self.config
        kwargs = self._decode_kwargs()

        from faster_whisper import decode_audio

        audio = decode_audio(str(audio_path), sampling_rate=SAMPLE_RATE)
        bounds = chunk_bounds(audio)
        segments: list[Segment] = []
        info = None
        for start, end in zip(bounds, bounds[1:], strict=False):
            offset = start / SAMPLE_RATE
            segment_iter, chunk_info = self._model.transcribe(audio[start:end], **kwargs)
            if info is None:
                info = chunk_info
                kwargs["language"] = info.language  # "auto": detect once, keep it for the rest
            for s in segment_iter:
                seg = Segment(
                    start=offset + float(s.start),
                    end=offset + float(s.end),
                    text=s.text.strip(),
                    confidence=segment_confidence(s),
                )
                segments.append(seg)
                if on_segment is not None:
                    on_segment(seg)
                if progress is not None:
                    progress(min(1.0, (offset + float(s.end)) * SAMPLE_RATE / max(len(audio), 1)))
            if progress is not None:
                progress(end / max(len(audio), 1))
        return Transcript(
            segments=segments,
            language=info.language if info else cfg.language,
            duration=len(audio) / SAMPLE_RATE,
            engine=self.name,
            model=cfg.model,
        )
