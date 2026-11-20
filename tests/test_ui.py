"""The web UI builds with the installed Gradio and transcribes like the CLI."""

import pytest

from goftar.core.pipeline import Pipeline
from goftar.interfaces.ui import copy_button_kwargs, transcribe_for_ui


def test_ui_builds_with_installed_gradio(fake_config):
    gr = pytest.importorskip("gradio")
    from goftar.interfaces.ui import build_ui

    assert isinstance(build_ui(Pipeline(fake_config)), gr.Blocks)


def test_copy_button_option_matches_gradio_version():
    class Gradio6:
        def __init__(self, buttons=None): ...

    class Gradio5:
        def __init__(self, show_copy_button=False): ...

    assert copy_button_kwargs(Gradio6) == {"buttons": ["copy"]}
    assert copy_button_kwargs(Gradio5) == {"show_copy_button": True}


def test_transcribe_for_ui_writes_the_download(fake_config, wav_file):
    text, path = transcribe_for_ui(Pipeline(fake_config), str(wav_file), "srt")
    assert text.startswith("سلام، حال")
    assert path.endswith("voice.srt")


def test_transcribe_for_ui_explains_bad_input(fake_config, tmp_path):
    with pytest.raises(ValueError, match="Upload or record"):
        transcribe_for_ui(Pipeline(fake_config), None, "txt")
    bad = tmp_path / "bad.ogg"
    bad.write_bytes(b"not audio" * 20)
    with pytest.raises(ValueError, match="could not decode bad.ogg"):
        transcribe_for_ui(Pipeline(fake_config), str(bad), "txt")
