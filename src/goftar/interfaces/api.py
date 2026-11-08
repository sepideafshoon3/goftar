"""Local REST API (FastAPI).

NOTE: deliberately no `from __future__ import annotations` here -- FastAPI resolves endpoint
annotations at runtime and the FastAPI names are imported inside `create_app`.
"""

import re
import tempfile
import threading
import uuid
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

from goftar import __version__
from goftar.core.config import Config
from goftar.core.pipeline import Pipeline
from goftar.errors import GoftarError
from goftar.export.formats import FORMATS, render

MAX_UPLOAD_BYTES = 200 * 1024 * 1024
MAX_JOBS_KEPT = 200


def create_app(config: Config | None = None, pipeline: Pipeline | None = None):
    from fastapi import FastAPI, File, HTTPException, UploadFile
    from fastapi.responses import PlainTextResponse

    pipe = pipeline or Pipeline(config or Config())
    app = FastAPI(
        title="Goftar",
        version=__version__,
        description="Offline Persian speech-to-text. Runs entirely on this machine.",
    )
    app.state.pipeline = pipe

    jobs: dict[str, dict[str, Any]] = {}
    jobs_lock = threading.Lock()
    executor = ThreadPoolExecutor(max_workers=1)  # one model -> one job at a time

    def save_upload(upload: UploadFile) -> Path:
        suffix = re.sub(r"[^A-Za-z0-9.]", "", Path(upload.filename or "").suffix)[:10] or ".bin"
        tmp = tempfile.NamedTemporaryFile(  # noqa: SIM115 - closed by `with tmp` below
            delete=False, suffix=suffix, prefix="goftar-upload-"
        )
        path = Path(tmp.name)
        size = 0
        try:
            with tmp:
                while chunk := upload.file.read(1024 * 1024):
                    size += len(chunk)
                    if size > MAX_UPLOAD_BYTES:
                        raise HTTPException(413, "file too large")
                    tmp.write(chunk)
        except BaseException:
            path.unlink(missing_ok=True)
            raise
        return path

    @app.get("/health")
    def health():
        return {
            "status": "ok",
            "version": __version__,
            "engine": pipe.config.engine,
            "model": pipe.config.model,
        }

    @app.post("/transcribe")
    def transcribe(file: UploadFile = File(...), format: str = "json"):
        """Synchronous transcription. `format`: json (default), txt, srt or vtt."""
        if format not in FORMATS:
            raise HTTPException(422, f"format must be one of {sorted(FORMATS)}")
        path = save_upload(file)
        try:
            result = pipe.transcribe(path)
        except GoftarError as exc:
            raise HTTPException(422, str(exc)) from exc
        finally:
            path.unlink(missing_ok=True)
        if format == "json":
            return result.to_dict()
        return PlainTextResponse(render(result, format))

    @app.post("/jobs", status_code=202)
    def create_job(file: UploadFile = File(...)):
        """Queue a transcription and poll GET /jobs/{id} (good for long files)."""
        path = save_upload(file)
        job_id = uuid.uuid4().hex
        with jobs_lock:
            jobs[job_id] = {"status": "queued"}
            while len(jobs) > MAX_JOBS_KEPT:
                jobs.pop(next(iter(jobs)))

        def work() -> None:
            with jobs_lock:
                jobs[job_id] = {"status": "running"}
            try:
                outcome: dict[str, Any] = {
                    "status": "done",
                    "result": pipe.transcribe(path).to_dict(),
                }
            except Exception as exc:
                outcome = {"status": "failed", "error": str(exc)}
            finally:
                path.unlink(missing_ok=True)
            with jobs_lock:
                jobs[job_id] = outcome

        executor.submit(work)
        return {"job_id": job_id}

    @app.get("/jobs/{job_id}")
    def get_job(job_id: str):
        with jobs_lock:
            job = jobs.get(job_id)
        if job is None:
            raise HTTPException(404, "unknown job id")
        return job

    return app
