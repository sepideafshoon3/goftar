import time

import pytest

pytest.importorskip("fastapi")
pytest.importorskip("httpx")
pytest.importorskip("multipart")

from fastapi.testclient import TestClient  # noqa: E402

from goftar.core.pipeline import Pipeline  # noqa: E402
from goftar.interfaces.api import create_app  # noqa: E402


@pytest.fixture
def client(fake_config):
    return TestClient(create_app(pipeline=Pipeline(fake_config)))


def test_health(client):
    body = client.get("/health").json()
    assert body["status"] == "ok"
    assert body["engine"] == "fake"


def test_transcribe_json(client, wav_file):
    with wav_file.open("rb") as fh:
        response = client.post("/transcribe", files={"file": ("voice.wav", fh, "audio/wav")})
    assert response.status_code == 200
    body = response.json()
    assert body["text"].startswith("سلام، حال")
    assert len(body["segments"]) == 2


@pytest.mark.parametrize("fmt", ["txt", "srt", "vtt"])
def test_transcribe_text_formats(client, wav_file, fmt):
    with wav_file.open("rb") as fh:
        response = client.post(f"/transcribe?format={fmt}", files={"file": ("v.wav", fh)})
    assert response.status_code == 200
    assert "سلام" in response.text


def test_transcribe_rejects_unknown_format(client, wav_file):
    with wav_file.open("rb") as fh:
        response = client.post("/transcribe?format=docx", files={"file": ("v.wav", fh)})
    assert response.status_code == 422


def test_job_lifecycle(client, wav_file):
    with wav_file.open("rb") as fh:
        created = client.post("/jobs", files={"file": ("v.wav", fh)})
    assert created.status_code == 202
    job_id = created.json()["job_id"]
    for _ in range(50):
        job = client.get(f"/jobs/{job_id}").json()
        if job["status"] in {"done", "failed"}:
            break
        time.sleep(0.1)
    assert job["status"] == "done"
    assert job["result"]["text"]


def test_unknown_job(client):
    assert client.get("/jobs/does-not-exist").status_code == 404
