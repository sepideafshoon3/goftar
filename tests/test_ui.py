"""The web UI builds with the installed Gradio, speaks Persian and transcribes like the CLI."""

import pytest

from goftar.core.pipeline import Pipeline
from goftar.errors import AudioError, EngineError
from goftar.interfaces.ui import (
    clock,
    copy_button_kwargs,
    launch_options,
    persian_error,
    transcribe_for_ui,
)


def test_ui_builds_with_installed_gradio(fake_config):
    gr = pytest.importorskip("gradio")
    from goftar.interfaces.ui import build_ui

    assert isinstance(build_ui(Pipeline(fake_config)), gr.Blocks)


def test_launch_options_keep_the_page_rtl_and_allow_long_files():
    opts = launch_options()
    assert "direction: rtl" in opts["css"]
    assert opts["max_file_size"] == "500mb"


def test_copy_button_option_matches_gradio_version():
    class Gradio6:
        def __init__(self, buttons=None): ...

    class Gradio5:
        def __init__(self, show_copy_button=False): ...

    assert copy_button_kwargs(Gradio6) == {"buttons": ["copy"]}
    assert copy_button_kwargs(Gradio5) == {"show_copy_button": True}


def test_transcribe_for_ui_returns_text_segments_and_all_downloads(fake_config, wav_file):
    r = transcribe_for_ui(Pipeline(fake_config), str(wav_file))
    assert r.text.startswith("سلام، حال")
    assert sorted(p.rsplit(".", 1)[1] for p in r.files) == ["srt", "txt", "vtt"]
    assert r.rows[0] == ["00:00", "00:01", "سلام، حال شما چطوره؟"]
    assert len(r.rows) == 2  # the blank segment is left out
    assert "مدت صدا: ۰۰:۰۳" in r.summary


def test_silence_says_no_speech(fake_config, wav_file, monkeypatch):
    from dataclasses import replace

    from tests.conftest import FakeEngine

    original = FakeEngine.transcribe
    monkeypatch.setattr(
        FakeEngine, "transcribe", lambda self, p: replace(original(self, p), segments=[])
    )
    r = transcribe_for_ui(Pipeline(fake_config), str(wav_file))
    assert r.text == "" and r.rows == []
    assert r.summary == "صحبتی در این صدا پیدا نشد."


def test_transcribe_for_ui_explains_bad_input_in_persian(fake_config, tmp_path):
    with pytest.raises(ValueError, match="اول صدا را ضبط"):
        transcribe_for_ui(Pipeline(fake_config), None)
    bad = tmp_path / "bad.ogg"
    bad.write_bytes(b"not audio" * 20)
    with pytest.raises(ValueError, match="«bad.ogg» خوانده نشد: فایل صوتی نیست یا خراب است"):
        transcribe_for_ui(Pipeline(fake_config), str(bad))


def test_persian_error_keeps_unknown_reasons_and_engine_errors():
    assert persian_error(AudioError("could not decode a.mp3: the file is empty")) == (
        "فایل «a.mp3» خوانده نشد: فایل خالی است."
    )
    assert persian_error(AudioError("could not decode a.mp3: odd codec")).endswith("odd codec.")
    assert persian_error(EngineError("CUDA broke")) == "مدل اجرا نشد: CUDA broke"


@pytest.mark.parametrize(("sec", "text"), [(0, "00:00"), (12.9, "00:12"), (3725, "1:02:05")])
def test_clock(sec, text):
    assert clock(sec) == text


def test_pipeline_passes_progress_only_when_asked(fake_config, wav_file):
    from tests.conftest import FakeEngine

    seen = []

    class Recording(FakeEngine):
        def transcribe(self, audio_path, progress=None):
            seen.append(progress)
            return super().transcribe(audio_path)

    fake_config.cache.enabled = False
    pipe = Pipeline(fake_config)
    pipe._engine = Recording(fake_config)

    def cb(fraction):
        pass

    pipe.transcribe(wav_file)
    pipe.transcribe(wav_file, progress=cb)
    assert seen == [None, cb]
