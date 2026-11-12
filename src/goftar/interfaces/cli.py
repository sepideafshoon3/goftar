"""Command-line interface (Typer)."""

import logging
import os
from pathlib import Path
from typing import Annotated, Any

import typer

from goftar import __version__
from goftar.audio.ingest import collect_audio_files
from goftar.core.config import DEFAULT_MODEL, Config, load_config
from goftar.core.pipeline import Pipeline
from goftar.errors import ConfigError, GoftarError
from goftar.export.formats import FORMATS, write_transcript

app = typer.Typer(
    help="Goftar: offline Persian speech-to-text.",
    no_args_is_help=True,
    add_completion=False,
)
models_app = typer.Typer(help="Manage local models.", no_args_is_help=True)
app.add_typer(models_app, name="models")
dataset_app = typer.Typer(
    help="Build the benchmark dataset: draft transcripts to correct by hand.",
    no_args_is_help=True,
)
app.add_typer(dataset_app, name="dataset")

ConfigOpt = Annotated[
    Path | None, typer.Option("--config", "-c", help="YAML config (default: ./goftar.yaml).")
]
ModelOpt = Annotated[str | None, typer.Option("--model", "-m", help="Model name or directory.")]
DeviceOpt = Annotated[str | None, typer.Option("--device", help="auto | cpu | cuda.")]
ComputeOpt = Annotated[
    str | None, typer.Option("--compute-type", help="int8, int8_float16, float16, ...")
]
LanguageOpt = Annotated[str | None, typer.Option("--language", "-l", help="'fa' or 'auto'.")]
OfflineOpt = Annotated[bool, typer.Option("--offline", help="Forbid all network access.")]
VerboseOpt = Annotated[bool, typer.Option("--verbose", "-v", help="Debug logging.")]


def _setup_logging(verbose: bool) -> None:
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.WARNING,
        format="%(levelname)s %(name)s: %(message)s",
    )


def _fail(message: str, code: int = 1) -> typer.Exit:
    typer.secho(message, fg=typer.colors.RED, err=True)
    return typer.Exit(code)


def _build_config(
    config: Path | None,
    *,
    model: str | None = None,
    device: str | None = None,
    compute_type: str | None = None,
    language: str | None = None,
    offline: bool = False,
    no_cache: bool = False,
    engine: str | None = None,
) -> Config:
    overrides: dict[str, Any] = {
        k: v
        for k, v in {
            "model": model,
            "device": device,
            "compute_type": compute_type,
            "language": language,
            "engine": engine,
        }.items()
        if v is not None
    }
    if offline:
        overrides["offline"] = True
    if no_cache:
        overrides["cache"] = {"enabled": False}
    try:
        return load_config(config, overrides)
    except ConfigError as exc:
        raise _fail(f"config error: {exc}", 2) from exc


@app.command()
def version() -> None:
    """Print the version."""
    typer.echo(__version__)


@app.command()
def transcribe(
    paths: Annotated[list[Path], typer.Argument(help="Audio files and/or folders.", exists=True)],
    out_dir: Annotated[
        Path | None, typer.Option("--out-dir", "-o", help="Default: next to each input file.")
    ] = None,
    formats: Annotated[str, typer.Option("--formats", "-f", help="txt,srt,vtt,json")] = "txt",
    recursive: Annotated[
        bool, typer.Option("--recursive", "-r", help="Recurse into folders.")
    ] = False,
    stdout: Annotated[bool, typer.Option("--stdout", help="Also print the text.")] = False,
    no_cache: Annotated[bool, typer.Option("--no-cache", help="Ignore the result cache.")] = False,
    config: ConfigOpt = None,
    model: ModelOpt = None,
    device: DeviceOpt = None,
    compute_type: ComputeOpt = None,
    language: LanguageOpt = None,
    offline: OfflineOpt = False,
    engine: Annotated[str | None, typer.Option("--engine", hidden=True)] = None,
    verbose: VerboseOpt = False,
) -> None:
    """Transcribe audio files or whole folders (batch)."""
    _setup_logging(verbose)
    fmts = [f.strip().lower() for f in formats.split(",") if f.strip()]
    bad = [f for f in fmts if f not in FORMATS]
    if bad or not fmts:
        raise _fail(f"unknown format(s) {bad or formats!r}; choose from {sorted(FORMATS)}", 2)

    files = collect_audio_files(paths, recursive=recursive)
    if not files:
        raise _fail("no supported audio files found")

    cfg = _build_config(
        config,
        model=model,
        device=device,
        compute_type=compute_type,
        language=language,
        offline=offline,
        no_cache=no_cache,
        engine=engine,
    )
    pipeline = Pipeline(cfg)

    failures = 0
    for audio, outcome in pipeline.transcribe_many(files):
        if isinstance(outcome, Exception):
            failures += 1
            typer.secho(f"FAILED {audio}: {outcome}", fg=typer.colors.RED, err=True)
            continue
        dest = out_dir or audio.parent
        written = write_transcript(outcome, dest, audio.stem, fmts)
        typer.echo(f"{audio.name} -> {', '.join(p.name for p in written)}", err=True)
        if stdout:
            typer.echo(outcome.text)
    if failures:
        raise _fail(f"{failures} of {len(files)} file(s) failed")


@app.command()
def serve(
    host: Annotated[str, typer.Option(help="Bind address (localhost by default).")] = "127.0.0.1",
    port: Annotated[int, typer.Option(help="Port.")] = 8000,
    config: ConfigOpt = None,
    model: ModelOpt = None,
    device: DeviceOpt = None,
    compute_type: ComputeOpt = None,
    language: LanguageOpt = None,
    offline: OfflineOpt = False,
    verbose: VerboseOpt = False,
) -> None:
    """Run the local REST API (needs `pip install 'goftar[api]'`)."""
    _setup_logging(verbose)
    try:
        import uvicorn

        from goftar.interfaces.api import create_app
    except ImportError as exc:
        raise _fail("API extras missing: pip install 'goftar[api]'") from exc
    cfg = _build_config(
        config, model=model, device=device, compute_type=compute_type,
        language=language, offline=offline,
    )  # fmt: skip
    uvicorn.run(create_app(cfg), host=host, port=port)


@app.command()
def ui(
    host: Annotated[str, typer.Option(help="Bind address (localhost by default).")] = "127.0.0.1",
    port: Annotated[int, typer.Option(help="Port.")] = 7860,
    config: ConfigOpt = None,
    model: ModelOpt = None,
    device: DeviceOpt = None,
    compute_type: ComputeOpt = None,
    language: LanguageOpt = None,
    offline: OfflineOpt = False,
    verbose: VerboseOpt = False,
) -> None:
    """Run the local web UI (needs `pip install 'goftar[ui]'`)."""
    _setup_logging(verbose)
    os.environ.setdefault("GRADIO_ANALYTICS_ENABLED", "False")  # stay offline/private
    try:
        from goftar.interfaces.ui import build_ui
    except ImportError as exc:
        raise _fail("UI extras missing: pip install 'goftar[ui]'") from exc
    cfg = _build_config(
        config, model=model, device=device, compute_type=compute_type,
        language=language, offline=offline,
    )  # fmt: skip
    build_ui(Pipeline(cfg)).launch(server_name=host, server_port=port)


@app.command("fix-text")
def fix_text(
    text: Annotated[str, typer.Argument(help="Persian text to clean.")],
    config: ConfigOpt = None,
    spelling: Annotated[
        bool, typer.Option("--spelling", help="Also fix homophone-letter typos.")
    ] = False,
) -> None:
    """Run the text clean-up (normalisation, glossary, optional spelling) on a string."""
    from goftar.text.normalize import clean_text

    cfg = _build_config(config)
    if spelling:
        cfg.postprocess.spelling = True
    try:
        typer.echo(clean_text(text, cfg.postprocess))
    except ConfigError as exc:
        raise _fail(f"config error: {exc}", 2) from exc


@app.command()
def doctor(config: ConfigOpt = None) -> None:
    """Check ffmpeg, GPU, models and optional extras."""
    from goftar.doctor import run_checks

    cfg = _build_config(config)
    failed_required = False
    for check in run_checks(cfg):
        mark = "ok  " if check.ok else ("FAIL" if check.required else "warn")
        color = (
            typer.colors.GREEN
            if check.ok
            else (typer.colors.RED if check.required else typer.colors.YELLOW)
        )
        typer.secho(f"[{mark}] {check.name}: {check.detail}", fg=color)
        failed_required |= check.required and not check.ok
    if failed_required:
        raise typer.Exit(1)


@app.command()
def bench(
    dataset: Annotated[Path, typer.Argument(help="Folder with audio + same-name .txt references.")],
    models: Annotated[str, typer.Option(help="Comma-separated model names.")] = DEFAULT_MODEL,
    out: Annotated[Path, typer.Option(help="Markdown report path.")] = Path("bench/results.md"),
    details: Annotated[
        bool,
        typer.Option(
            "--details",
            help="Also write <out>.details.md: per-file scores, frequent errors, worst samples. "
            "Contains your transcript text, so it is git-ignored.",
        ),
    ] = False,
    top: Annotated[int, typer.Option(help="Rows in the frequent-errors table of --details.")] = 25,
    config: ConfigOpt = None,
    device: DeviceOpt = None,
    compute_type: ComputeOpt = None,
    verbose: VerboseOpt = False,
) -> None:
    """Benchmark models on your own labelled audio (WER, CER, speed, RAM)."""
    _setup_logging(verbose)
    from goftar.bench import discover_samples, render_details, render_markdown, run_benchmark

    samples = discover_samples(dataset)
    if not samples:
        raise _fail(f"no audio+.txt pairs found in {dataset}")
    configs = {
        name.strip(): _build_config(config, model=name.strip(), device=device,
                                    compute_type=compute_type)
        for name in models.split(",")
        if name.strip()
    }  # fmt: skip
    results = run_benchmark(samples, configs)
    report = render_markdown(results, n_samples=len(samples))
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(report, encoding="utf-8")
    typer.echo(report)
    typer.echo(f"saved to {out}", err=True)
    if details:
        details_path = out.with_suffix(".details.md")
        details_path.write_text(render_details(results, top=top), encoding="utf-8")
        typer.echo(f"details saved to {details_path} (personal data, git-ignored)", err=True)


@models_app.command("download")
def models_download(
    name: Annotated[
        str,
        typer.Argument(help="tiny, base, small, medium, large-v3, large-v3-turbo, or HF repo id."),
    ] = DEFAULT_MODEL,
    models_dir: Annotated[
        Path | None, typer.Option(help="Default: models_dir from config.")
    ] = None,
    config: ConfigOpt = None,
) -> None:
    """Download a model once (needs internet) so it can be used fully offline afterwards."""
    from goftar.engines.faster_whisper_engine import download_model

    cfg = _build_config(config)
    try:
        target = download_model(name, models_dir or cfg.models_dir)
    except GoftarError as exc:
        raise _fail(str(exc)) from exc
    typer.echo(f"model saved to {target}")


@models_app.command("list")
def models_list(
    models_dir: Annotated[
        Path | None, typer.Option(help="Default: models_dir from config.")
    ] = None,
    config: ConfigOpt = None,
) -> None:
    """List models available locally."""
    cfg = _build_config(config)
    root = (models_dir or Path(cfg.models_dir)).expanduser()
    found = (
        sorted(p for p in root.glob("*") if (p / "model.bin").is_file()) if root.is_dir() else []
    )
    if not found:
        typer.echo(f"no models in {root} (try: goftar models download large-v3-turbo)")
        return
    for p in found:
        size_mb = sum(f.stat().st_size for f in p.rglob("*") if f.is_file()) / 1e6
        typer.echo(f"{p.name}\t{size_mb:.0f} MB")


@dataset_app.command("prepare")
def dataset_prepare(
    folder: Annotated[Path, typer.Argument(exists=True, file_okay=False, help="Dataset folder.")],
    overwrite: Annotated[
        bool, typer.Option("--overwrite", help="Regenerate existing drafts (never references).")
    ] = False,
    config: ConfigOpt = None,
    model: ModelOpt = None,
    device: DeviceOpt = None,
    compute_type: ComputeOpt = None,
    language: LanguageOpt = None,
    offline: OfflineOpt = False,
    engine: Annotated[str | None, typer.Option("--engine", hidden=True)] = None,
    verbose: VerboseOpt = False,
) -> None:
    """Write <name>.draft.txt next to each audio file that has no reference yet."""
    _setup_logging(verbose)
    from goftar.dataset import FAILED, WRITTEN, prepare_drafts

    cfg = _build_config(
        config,
        model=model,
        device=device,
        compute_type=compute_type,
        language=language,
        offline=offline,
        no_cache=True,
        engine=engine,
    )
    outcomes = prepare_drafts(Pipeline(cfg), folder, overwrite=overwrite)
    for o in outcomes:
        detail = f" ({o.detail})" if o.detail else ""
        typer.echo(f"{o.status:14} {o.audio.name}{detail}")
    written = sum(o.status == WRITTEN for o in outcomes)
    typer.echo(
        f"\n{written} draft(s) written. Listen, correct EVERY word, then run:\n"
        f"  goftar dataset accept {folder}/<audio-file> ...",
        err=True,
    )
    if any(o.status == FAILED for o in outcomes):
        raise typer.Exit(1)


@dataset_app.command("status")
def dataset_status(
    folder: Annotated[Path, typer.Argument(exists=True, file_okay=False, help="Dataset folder.")],
) -> None:
    """Show which audio files are ready, awaiting review, or still missing a transcript."""
    from goftar.dataset import DRAFT, MISSING, READY, list_entries

    entries = list_entries(folder)
    if not entries:
        raise _fail(f"no audio files in {folder}")
    marks = {READY: "ok   ", DRAFT: "review", MISSING: "todo "}
    for e in entries:
        typer.echo(f"[{marks[e.state]}] {e.audio.name}")
    counts = {s: sum(e.state == s for e in entries) for s in (READY, DRAFT, MISSING)}
    typer.echo(
        f"\n{counts[READY]} ready, {counts[DRAFT]} awaiting review, {counts[MISSING]} without text"
        f"  (benchmark uses the {counts[READY]} ready ones)"
    )


@dataset_app.command("accept")
def dataset_accept(
    audio: Annotated[list[Path], typer.Argument(help="Audio file(s) whose draft you corrected.")],
    force: Annotated[bool, typer.Option("--force", help="Replace an existing reference.")] = False,
) -> None:
    """Turn corrected drafts into references (<name>.draft.txt -> <name>.txt)."""
    from goftar.dataset import accept_draft

    failed = False
    for path in audio:
        try:
            ref = accept_draft(path, force=force)
        except (FileNotFoundError, FileExistsError) as exc:
            failed = True
            typer.secho(f"skipped {path.name}: {exc}", fg=typer.colors.RED, err=True)
            continue
        typer.echo(f"accepted {ref.name}")
    if failed:
        raise typer.Exit(1)


if __name__ == "__main__":
    app()
