from __future__ import annotations

import pytest

from goftar.core.config import DEFAULT_MODEL, Config
from goftar.engines import faster_whisper_engine as fw


def test_default_model_is_turbo():
    assert DEFAULT_MODEL == "large-v3-turbo"
    assert Config().model == DEFAULT_MODEL


@pytest.mark.parametrize(
    ("gpu", "expected"),
    [(True, ("cuda", "int8_float16")), (False, ("cpu", "int8"))],
)
def test_auto_resolves_per_device(monkeypatch, gpu, expected):
    monkeypatch.setattr(fw, "cuda_available", lambda: gpu)
    assert fw.resolve_device("auto", "auto") == expected


def test_auto_compute_follows_explicit_device(monkeypatch):
    monkeypatch.setattr(fw, "cuda_available", lambda: True)
    assert fw.resolve_device("cpu", "auto") == ("cpu", "int8")
    assert fw.resolve_device("cuda", "auto") == ("cuda", "int8_float16")


def test_explicit_values_pass_through(monkeypatch):
    monkeypatch.setattr(fw, "cuda_available", lambda: False)
    assert fw.resolve_device("cuda", "float16") == ("cuda", "float16")
    assert fw.resolve_device("auto", "float32") == ("cpu", "float32")
