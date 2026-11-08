"""Environment diagnostics: `goftar doctor`."""

from __future__ import annotations

import importlib.metadata
import importlib.util
import os
import platform
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path

from goftar.core.config import Config


@dataclass
class Check:
    name: str
    ok: bool
    detail: str
    required: bool = False


def _version(dist: str) -> str | None:
    try:
        return importlib.metadata.version(dist)
    except importlib.metadata.PackageNotFoundError:
        return None


def _run(cmd: list[str]) -> str | None:
    try:
        out = subprocess.run(cmd, capture_output=True, text=True, timeout=15, check=False)
    except (OSError, subprocess.TimeoutExpired):
        return None
    return out.stdout.strip() if out.returncode == 0 else None


def _cuda_devices() -> int | None:
    if importlib.util.find_spec("ctranslate2") is None:
        return None
    try:
        import ctranslate2

        return int(ctranslate2.get_cuda_device_count())
    except Exception:
        return None


def _models_in(models_dir: Path) -> list[str]:
    if not models_dir.is_dir():
        return []
    return sorted(p.name for p in models_dir.iterdir() if (p / "model.bin").is_file())


def run_checks(config: Config | None = None) -> list[Check]:
    config = config or Config()
    checks = [Check("python", True, platform.python_version())]

    ffmpeg = shutil.which("ffmpeg")
    first_line = (_run(["ffmpeg", "-version"]) or "").splitlines()[:1]
    checks.append(
        Check("ffmpeg", ffmpeg is not None, first_line[0] if first_line else "not found", True)
    )

    fw = _version("faster-whisper")
    checks.append(Check("faster-whisper", fw is not None, fw or "not installed", True))

    devices = _cuda_devices()
    if devices is None:
        checks.append(Check("cuda (ctranslate2)", False, "ctranslate2 not available"))
    else:
        checks.append(Check("cuda (ctranslate2)", devices > 0, f"{devices} device(s) visible"))

    gpu = _run(["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"])
    checks.append(Check("nvidia-smi", gpu is not None, gpu or "no NVIDIA driver found"))

    for label, dist in [("fastapi", "fastapi"), ("gradio", "gradio"), ("hazm", "hazm")]:
        v = _version(dist)
        checks.append(Check(label + " (optional)", v is not None, v or "not installed"))

    models = _models_in(Path(config.models_dir).expanduser())
    checks.append(
        Check("local models", bool(models), ", ".join(models) or f"none in ./{config.models_dir}")
    )
    offline = os.environ.get("HF_HUB_OFFLINE") == "1" or config.offline
    checks.append(Check("offline mode", True, "on" if offline else "off"))
    return checks
