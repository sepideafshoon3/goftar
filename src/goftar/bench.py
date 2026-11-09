"""Benchmark harness: WER / CER / real-time factor / memory per model on your own audio.

Dataset layout: a folder of audio files, each with a reference transcript next to it:
    voice1.ogg + voice1.txt
"""

from __future__ import annotations

import gc
import platform
import time
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

from goftar.audio.ingest import SUPPORTED_EXTENSIONS
from goftar.core.config import Config
from goftar.core.pipeline import Pipeline
from goftar.metrics import cer, error_spans, tokenize, wer, word_errors


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
    reference: str = ""
    hypothesis: str = ""


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
                        reference=sample.reference,
                        hypothesis=tr.text,
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
        f"- Samples: {n_samples} (hand-checked reference transcripts)",
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


NOTHING = "-"


@dataclass
class SharedErrors:
    """Reference words that EVERY model got wrong."""

    labels: list[str]
    rows: list[tuple[str, str, list[str]]]  # (file, reference words, what each model heard)
    shared_words: int
    fewest_wrong_words: int  # total wrong words of the best model (the share's denominator)


def shared_errors(results: list[ModelResult]) -> SharedErrors | None:
    """Find reference words wrong in every model: accent/audio problems or reference typos."""
    ok = [r for r in results if not r.error and r.samples]
    if len(ok) < 2:
        return None
    by_model = [{s.name: s for s in r.samples} for r in ok]
    names = sorted(set.intersection(*(set(d) for d in by_model)))

    rows: list[tuple[str, str, list[str]]] = []
    shared_words = 0
    wrong_totals = [0] * len(ok)
    for name in names:
        ref_tokens = tokenize(by_model[0][name].reference)
        spans = [error_spans(d[name].reference, d[name].hypothesis) for d in by_model]
        wrong = [
            {i for sp in model_spans for i in range(sp.start, sp.end)} for model_spans in spans
        ]
        for k, indices in enumerate(wrong):
            wrong_totals[k] += len(indices)
        common = sorted(set.intersection(*wrong))
        shared_words += len(common)

        runs: list[list[int]] = []
        for i in common:
            if runs and i == runs[-1][-1] + 1:
                runs[-1].append(i)
            else:
                runs.append([i])
        for run in runs:
            heard = []
            for model_spans in spans:
                hit = next((sp for sp in model_spans if sp.start <= run[0] < sp.end), None)
                heard.append(hit.heard if hit and hit.heard else NOTHING)
            rows.append((name, " ".join(ref_tokens[run[0] : run[-1] + 1]), heard))
    return SharedErrors([r.label for r in ok], rows, shared_words, min(wrong_totals))


def render_details(
    results: list[ModelResult], *, top: int = 25, worst: int = 3, shared_rows: int = 60
) -> str:
    """Per-file scores, the most frequent word errors and the worst samples, per model.

    This contains transcript text from your own recordings: keep it local, do not commit it.
    """
    lines = [
        "# Benchmark details",
        "",
        "> Contains text from your recordings (personal data). Keep it local, do not commit.",
        "",
    ]
    for r in results:
        lines += [f"## {r.label} ({r.device}, {r.compute_type})", ""]
        if r.error:
            lines += [f"error: {r.error}", ""]
            continue

        lines += ["| File | Audio (s) | WER | CER |", "|---|---|---|---|"]
        ranked = sorted(r.samples, key=lambda s: s.wer, reverse=True)
        for s in ranked:
            lines.append(f"| {s.name} | {s.duration:.0f} | {s.wer:.1%} | {s.cer:.1%} |")

        errors: Counter[tuple[str, str]] = Counter()
        for s in r.samples:
            errors.update(word_errors(s.reference, s.hypothesis))
        lines += [
            "",
            "### Most frequent word errors",
            "",
            "| Count | Reference | Heard |",
            "|---|---|---|",
        ]
        for (ref, heard), count in errors.most_common(top):
            lines.append(f"| {count} | {ref or NOTHING} | {heard or NOTHING} |")

        lines += ["", f"### {min(worst, len(ranked))} worst samples", ""]
        for s in ranked[:worst]:
            lines += [
                f"**{s.name}** (WER {s.wer:.1%})",
                "",
                f"- reference: {s.reference}",
                f"- heard: {s.hypothesis}",
                "",
            ]
    shared = shared_errors(results)
    if shared is not None:
        share = shared.shared_words / shared.fewest_wrong_words if shared.fewest_wrong_words else 0
        lines += [
            "## Errors shared by every model",
            "",
            f"{shared.shared_words} reference words were wrong in every model. The best model got "
            f"{shared.fewest_wrong_words} words wrong in total, so about {share:.0%} of its errors "
            "are shared. A high share means a bigger or different model will not fix them: the "
            "speech is hard for all of them (accent, audio) or the reference has a typo. "
            "Listen to these first.",
            "",
            "| File | Reference | " + " | ".join(f"Heard: {m}" for m in shared.labels) + " |",
            "|---|---|" + "---|" * len(shared.labels),
        ]
        for name, words, heard in shared.rows[:shared_rows]:
            lines.append(f"| {name} | {words} | " + " | ".join(heard) + " |")
        if len(shared.rows) > shared_rows:
            lines.append(f"\n({len(shared.rows) - shared_rows} more not shown)")
        lines.append("")
    return "\n".join(lines) + "\n"
