"""faster-whisper (CTranslate2) backend."""

from __future__ import annotations

import logging
import math
import os
from pathlib import Path
from typing import Any

from goftar.core.config import AUTO_COMPUTE_TYPE, Config
from goftar.core.types import Segment, Transcript
from goftar.engines.base import ASREngine
from goftar.errors import EngineError

log = logging.getLogger(__name__)


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

    def transcribe(self, audio_path: str | Path) -> Transcript:
        self.load()
        cfg = self.config
        kwargs: dict[str, Any] = {
            "language": None if cfg.language == "auto" else cfg.language,
            "beam_size": cfg.beam_size,
            "vad_filter": cfg.vad.enabled,
            "condition_on_previous_text": cfg.condition_on_previous_text,
            "initial_prompt": cfg.initial_prompt or None,
        }
        if cfg.vad.enabled:
            kwargs["vad_parameters"] = {"min_silence_duration_ms": cfg.vad.min_silence_ms}

        segment_iter, info = self._model.transcribe(str(audio_path), **kwargs)
        segments = [
            Segment(
                start=float(s.start),
                end=float(s.end),
                text=s.text.strip(),
                confidence=math.exp(s.avg_logprob) if s.avg_logprob is not None else None,
            )
            for s in segment_iter
        ]
        return Transcript(
            segments=segments,
            language=info.language,
            duration=float(info.duration),
            engine=self.name,
            model=cfg.model,
        )
