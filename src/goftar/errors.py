"""Exception hierarchy. Everything Goftar raises on purpose derives from GoftarError."""

from __future__ import annotations


class GoftarError(Exception):
    """Base class for all expected Goftar errors."""


class ConfigError(GoftarError, ValueError):
    """Invalid or unreadable configuration."""


class AudioError(GoftarError):
    """The audio file could not be decoded or prepared."""


class EngineError(GoftarError):
    """The ASR engine is missing, misconfigured or failed."""
