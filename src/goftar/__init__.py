"""Goftar: offline Persian speech-to-text."""

import os

__version__ = "0.1.0"

# onnxruntime (used by faster-whisper's Silero VAD) sends usage telemetry to Microsoft
# (mobile.events.data.microsoft.com, with a persistent device ID) about 15 s into a run, even on
# Linux. onnxruntime.disable_telemetry_events() does not stop it; this variable does. It must
# be set before onnxruntime starts, so it lives here, at the top of the package. Goftar promises
# that nothing leaves the machine, so it is always on. Set it to 0 yourself to opt back in.
os.environ.setdefault("ORT_DISABLE_TELEMETRY", "1")

from goftar.core.config import Config, load_config  # noqa: E402
from goftar.core.pipeline import Pipeline  # noqa: E402
from goftar.core.types import Segment, Transcript  # noqa: E402

__all__ = ["Config", "Pipeline", "Segment", "Transcript", "__version__", "load_config"]
