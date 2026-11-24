"""Pipeline orchestration: ingest -> ASR -> post-process -> (cache)."""

from __future__ import annotations

import inspect
import logging
import threading
import time
from collections.abc import Iterable, Iterator
from dataclasses import replace
from pathlib import Path

from goftar.audio.ingest import prepared_audio
from goftar.core.cache import TranscriptCache
from goftar.core.config import Config, apply_offline
from goftar.core.types import Segment, Transcript
from goftar.engines import ASREngine, get_engine
from goftar.engines.base import ProgressFn, SegmentFn
from goftar.errors import EngineError, GoftarError
from goftar.text.normalize import clean_text, postprocess_transcript

log = logging.getLogger(__name__)


def _accepts(fn, name: str) -> bool:
    params = inspect.signature(fn).parameters
    return name in params or any(p.kind is p.VAR_KEYWORD for p in params.values())


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

    def _clean(self, seg: Segment) -> Segment:
        """One segment through the same post-processing as a whole transcript."""
        if not self.config.postprocess.enabled:
            return seg
        return replace(seg, text=clean_text(seg.text, self.config.postprocess))

    def transcribe_samples(self, samples, *, fast: bool = False) -> list[Segment]:
        """Live audio held in memory (16 kHz mono float32): cleaned, non-empty segments."""
        with self._lock:
            try:
                raw = self.engine.transcribe_samples(samples, fast=fast)
            except NotImplementedError as exc:
                raise EngineError(str(exc)) from exc
        return [s for s in map(self._clean, raw) if s.text]

    def transcribe(
        self,
        path: str | Path,
        progress: ProgressFn | None = None,
        on_segment: SegmentFn | None = None,
    ) -> Transcript:
        """Transcribe one file.

        `on_segment` gets each cleaned segment as soon as it is ready (not on a cache hit:
        the whole result comes back at once then).
        """
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
            # only pass the callbacks when asked for, so engines without them keep working
            kwargs: dict = {}
            if progress is not None and _accepts(self.engine.transcribe, "progress"):
                kwargs["progress"] = progress
            streams = on_segment is not None and _accepts(self.engine.transcribe, "on_segment")
            if streams:

                def emit(seg: Segment) -> None:
                    cleaned = self._clean(seg)
                    if cleaned.text:
                        on_segment(cleaned)

                kwargs["on_segment"] = emit
            raw = self.engine.transcribe(audio, **kwargs)
        if on_segment is not None and not streams:  # the engine cannot stream: all at the end
            for seg in postprocess_transcript(raw, self.config.postprocess).segments:
                if seg.text.strip():
                    on_segment(seg)
        result = postprocess_transcript(raw, self.config.postprocess)
        result = replace(result, processing_time=time.perf_counter() - started)

        if cache is not None and key is not None:
            cache.put(key, result)
        return result

    def transcribe_many(
        self, paths: Iterable[str | Path]
    ) -> Iterator[tuple[Path, Transcript | Exception]]:
        """Transcribe several files; a bad file never stops the rest.

        An EngineError (model missing, CUDA broken) is raised instead: it would fail every
        file the same way, so the batch stops at once.
        """
        for p in paths:
            try:
                yield Path(p), self.transcribe(p)
            except EngineError:
                raise
            except GoftarError as exc:  # expected (bad file): the caller reports it in one line
                log.debug("failed to transcribe %s", p, exc_info=True)
                yield Path(p), exc
            except Exception as exc:
                log.exception("failed to transcribe %s", p)
                yield Path(p), exc
