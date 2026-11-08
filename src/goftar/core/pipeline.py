"""Pipeline orchestration: ingest -> ASR -> post-process -> (cache)."""

from __future__ import annotations

import logging
import threading
import time
from collections.abc import Iterable, Iterator
from dataclasses import replace
from pathlib import Path

from goftar.audio.ingest import prepared_audio
from goftar.core.cache import TranscriptCache
from goftar.core.config import Config, apply_offline
from goftar.core.types import Transcript
from goftar.engines import ASREngine, get_engine
from goftar.text.normalize import postprocess_transcript

log = logging.getLogger(__name__)


class Pipeline:
    """Turns an audio file into a clean :class:`Transcript`.

    The engine is created lazily, so building a Pipeline is cheap and never loads a model.
    Calls are serialised with a lock: one loaded model, one transcription at a time.
    """

    def __init__(
        self,
        config: Config | None = None,
        *,
        engine: ASREngine | None = None,
        cache: TranscriptCache | None = None,
    ) -> None:
        self.config = config or Config()
        apply_offline(self.config)
        self._engine = engine
        self._cache = cache
        self._cache_ready = cache is not None
        self._lock = threading.Lock()

    @property
    def engine(self) -> ASREngine:
        if self._engine is None:
            self._engine = get_engine(self.config)
        return self._engine

    def _get_cache(self) -> TranscriptCache | None:
        if not self._cache_ready:
            self._cache_ready = True
            if self.config.cache.enabled:
                self._cache = TranscriptCache(self.config.cache.path)
        return self._cache

    def transcribe(self, path: str | Path) -> Transcript:
        audio_path = Path(path).expanduser()
        if not audio_path.is_file():
            raise FileNotFoundError(f"audio file not found: {audio_path}")

        cache = self._get_cache()
        key = cache.make_key(audio_path, self.config.fingerprint()) if cache else None
        if cache is not None and key is not None:
            hit = cache.get(key)
            if hit is not None:
                log.info("cache hit for %s", audio_path.name)
                return hit

        started = time.perf_counter()
        with prepared_audio(audio_path, self.config.audio) as audio, self._lock:
            raw = self.engine.transcribe(audio)
        result = postprocess_transcript(raw, self.config.postprocess)
        result = replace(result, processing_time=time.perf_counter() - started)

        if cache is not None and key is not None:
            cache.put(key, result)
        return result

    def transcribe_many(
        self, paths: Iterable[str | Path]
    ) -> Iterator[tuple[Path, Transcript | Exception]]:
        """Transcribe several files; a failure on one file never stops the rest."""
        for p in paths:
            try:
                yield Path(p), self.transcribe(p)
            except Exception as exc:
                log.exception("failed to transcribe %s", p)
                yield Path(p), exc
