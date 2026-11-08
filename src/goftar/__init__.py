"""Goftar: offline Persian speech-to-text."""

__version__ = "0.1.0"

from goftar.core.config import Config, load_config  # noqa: E402
from goftar.core.pipeline import Pipeline  # noqa: E402
from goftar.core.types import Segment, Transcript  # noqa: E402

__all__ = ["Config", "Pipeline", "Segment", "Transcript", "__version__", "load_config"]
