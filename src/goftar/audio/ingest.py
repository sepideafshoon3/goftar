"""Audio ingestion: decode anything FFmpeg understands into 16 kHz mono WAV."""

from __future__ import annotations

import logging
import re
import shutil
import subprocess
import tempfile
from collections.abc import Iterable, Iterator
from contextlib import contextmanager
from pathlib import Path

from goftar.core.config import AudioConfig
from goftar.errors import AudioError

log = logging.getLogger(__name__)

SUPPORTED_EXTENSIONS = {
    ".wav", ".mp3", ".ogg", ".oga", ".opus", ".m4a", ".aac", ".flac",
    ".mp4", ".webm", ".mkv", ".wma", ".amr",
}  # fmt: skip

DENOISE_FILTER = "highpass=f=80,afftdn=nf=-25"


def ffmpeg_available() -> bool:
    return shutil.which("ffmpeg") is not None


def build_ffmpeg_cmd(src: Path, dst: Path, cfg: AudioConfig) -> list[str]:
    cmd = [
        "ffmpeg", "-nostdin", "-y", "-loglevel", "error",
        "-i", str(src),
        "-vn", "-ac", "1", "-ar", str(cfg.sample_rate),
    ]  # fmt: skip
    if cfg.denoise:
        cmd += ["-af", DENOISE_FILTER]
    cmd += ["-c:a", "pcm_s16le", str(dst)]
    return cmd


@contextmanager
def prepared_audio(path: Path, cfg: AudioConfig) -> Iterator[Path]:
    """Yield a path to engine-ready audio; temporary files are cleaned up on exit.

    Without FFmpeg we fall back to handing the original file to the engine (faster-whisper can
    decode most formats itself) -- but denoising needs FFmpeg, so that case is an error.
    """
    if not ffmpeg_available():
        if cfg.denoise:
            raise AudioError("audio.denoise requires ffmpeg, which was not found on PATH")
        log.warning("ffmpeg not found; passing %s to the engine undecoded", path.name)
        yield path
        return

    with tempfile.TemporaryDirectory(prefix="goftar-") as tmp:
        dst = Path(tmp) / "audio.wav"
        proc = subprocess.run(
            build_ffmpeg_cmd(path, dst, cfg), capture_output=True, text=True, check=False
        )
        if proc.returncode != 0 or not dst.exists():
            log.debug("ffmpeg failed on %s:\n%s", path, proc.stderr)
            raise AudioError(
                f"could not decode {path.name}: {explain_ffmpeg_error(path, proc.stderr)}"
            )
        yield dst


# (pattern in ffmpeg's stderr, message for people). First match wins.
_FFMPEG_ERRORS = [
    ("does not contain any stream", "it has no audio track"),
    ("matches no streams", "it has no audio track"),
    ("Invalid data found", "it is not an audio file, or it is damaged"),
    ("End of file", "it is not an audio file, or it is damaged"),
    ("moov atom not found", "the file is incomplete (an unfinished download or recording?)"),
    ("Permission denied", "permission denied"),
]
_FFMPEG_PREFIX = re.compile(r"^\[[^\]]*@ 0x[0-9a-f]+\]\s*")


def explain_ffmpeg_error(path: Path, stderr: str | None) -> str:
    """Turn FFmpeg's stderr into one short line (the full text is logged at debug level)."""
    if path.is_file() and path.stat().st_size == 0:
        return "the file is empty"
    stderr = stderr or ""
    for pattern, message in _FFMPEG_ERRORS:
        if pattern in stderr:
            return message
    lines = [_FFMPEG_PREFIX.sub("", ln).strip() for ln in stderr.splitlines() if ln.strip()]
    return lines[-1][:200] if lines else "unknown ffmpeg error"


def collect_audio_files(paths: Iterable[Path], recursive: bool = False) -> list[Path]:
    """Expand files/folders into a sorted, de-duplicated list of supported audio files."""
    found: dict[Path, None] = {}
    for p in paths:
        p = Path(p)
        if p.is_dir():
            pattern = "**/*" if recursive else "*"
            for child in sorted(p.glob(pattern)):
                if child.is_file() and child.suffix.lower() in SUPPORTED_EXTENSIONS:
                    found[child] = None
        elif p.is_file():
            found[p] = None
    return sorted(found)
