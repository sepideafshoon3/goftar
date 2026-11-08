"""ASR engine interface and registry.

Engines are registered by name. Built-in engines are imported lazily so that heavy
dependencies (CTranslate2, torch, ...) are only loaded when the engine is actually used.
Third parties and tests can plug in their own with :func:`register_engine`.
"""

from __future__ import annotations

import importlib
from collections.abc import Callable

from goftar.core.config import Config
from goftar.engines.base import ASREngine
from goftar.errors import EngineError

_BUILTIN: dict[str, str] = {
    "faster_whisper": "goftar.engines.faster_whisper_engine:FasterWhisperEngine",
}
_CUSTOM: dict[str, Callable[[Config], ASREngine]] = {}


def register_engine(name: str, factory: Callable[[Config], ASREngine]) -> None:
    _CUSTOM[name] = factory


def available_engines() -> list[str]:
    return sorted({*_BUILTIN, *_CUSTOM})


def get_engine(config: Config) -> ASREngine:
    name = config.engine
    if name in _CUSTOM:
        return _CUSTOM[name](config)
    if name in _BUILTIN:
        module_name, _, class_name = _BUILTIN[name].partition(":")
        cls = getattr(importlib.import_module(module_name), class_name)
        return cls(config)
    raise EngineError(f"unknown engine {name!r}; available: {', '.join(available_engines())}")


__all__ = ["ASREngine", "available_engines", "get_engine", "register_engine"]
