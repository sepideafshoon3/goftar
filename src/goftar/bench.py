"""Benchmark harness: WER / CER / real-time factor / memory per model on your own audio.

Dataset layout: a folder of audio files, each with a reference transcript next to it:
    voice1.ogg + voice1.txt
"""

from __future__ import annotations

import gc
import platform
import time
from dataclasses import dataclass, field
from pathlib import Path

from goftar.audio.ingest import SUPPORTED_EXTENSIONS
from goftar.core.config import Config
from goftar.core.pipeline import Pipeline
from goftar.metrics import cer, wer


@dataclass
class Sample:
    audio: Path
    reference: str


@dataclass
class SampleResult:
    name: str
    duration: float
    processing_time: float
    wer: float
    cer: float


@dataclass
class ModelResult:
    label: str
    device: str
    compute_type: str
    load_time: float = 0.0
    rss_mb: float | None = None
    samples: list[SampleResult] = field(default_factory=list)
    error: str | None = None

    @property
    def mean_wer(self) -> float:
        return sum(s.wer for s in self.samples) / len(self.samples) if self.samples else 0.0

    @property
    def mean_cer(self) -> float:
        return sum(s.cer for s in self.samples) / len(self.samples) if self.samples else 0.0

    @property
    def rtf(self) -> float:
        """Real-time factor: processing seconds per audio second (lower is faster)."""
        audio = sum(s.duration for s in self.samples)
        return sum(s.processing_time for s in self.samples) / audio if audio else 0.0


def discover_samples(dataset_dir: str | Path) -> list[Sample]:
    samples = []
    for audio in sorted(Path(dataset_dir).iterdir()):
        if audio.suffix.lower() not in SUPPORTED_EXTENSIONS:
            continue
        ref = audio.with_suffix(".txt")
        if ref.is_file():
            samples.append(Sample(audio, ref.read_text(encoding="utf-8").strip()))
    return samples


def _rss_mb() -> float | None:
    try:
        for line in Path("/proc/self/status").read_text().splitlines():
            if line.startswith("VmRSS:"):
                return int(line.split()[1]) / 1024
    except OSError:
        pass
    return None


def run_benchmark(samples: list[Sample], configs: dict[str, Config]) -> list[ModelResult]:
    results = []
    for label, cfg in configs.items():
        cfg.cache.enabled = False  # never benchmark the cache
        result = ModelResult(label=label, device=cfg.device, compute_type=cfg.compute_type)
        pipe = Pipeline(cfg)
        try:
            t0 = time.perf_counter()
            pipe.engine.load()
            result.load_time = time.perf_counter() - t0
            for sample in samples:
                tr = pipe.transcribe(sample.audio)
                duration = tr.duration or (tr.segments[-1].end if tr.segments else 0.0)
                result.samples.append(
                    SampleResult(
                        name=sample.audio.name,
                        duration=duration,
                        processing_time=tr.processing_time or 0.0,
                        wer=wer(sample.reference, tr.text),
                        cer=cer(sample.reference, tr.text),
                    )
                )
            result.rss_mb = _rss_mb()
        except Exception as exc:
            result.error = f"{type(exc).__name__}: {exc}"
        results.append(result)
        del pipe
        gc.collect()
    return results


def render_markdown(results: list[ModelResult], *, n_samples: int) -> str:
    lines = [
        "# Benchmark results",
        "",
        f"- Machine: {platform.platform()} / {platform.processor() or platform.machine()}",
        f"- Samples: {n_samples} (reference transcripts typed by hand)",
        "- WER/CER computed after Persian normalisation (punctuation and half-space ignored)",
        "- RTF = processing time / audio duration (lower is faster; < 1 is faster than real time)",
        "",
        "| Model | Device | Compute | WER | CER | RTF | Load (s) | RAM (MB) |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for r in results:
        if r.error:
            lines.append(f"| {r.label} | {r.device} | {r.compute_type} | error: {r.error} |||||")
            continue
        ram = f"{r.rss_mb:.0f}" if r.rss_mb is not None else "n/a"
        lines.append(
            f"| {r.label} | {r.device} | {r.compute_type} | {r.mean_wer:.1%} | {r.mean_cer:.1%} "
            f"| {r.rtf:.2f} | {r.load_time:.1f} | {ram} |"
        )
    return "\n".join(lines) + "\n"
