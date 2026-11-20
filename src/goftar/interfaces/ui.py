"""Local web UI (Gradio). Upload or record audio, get RTL text and a downloadable file."""

from __future__ import annotations

import inspect
import tempfile
from pathlib import Path
from typing import Any

from goftar.core.pipeline import Pipeline
from goftar.errors import GoftarError
from goftar.export.formats import FORMATS, write_transcript


def copy_button_kwargs(component: Any) -> dict[str, Any]:
    """The copy-button option for this Gradio version (renamed in Gradio 6)."""
    params = inspect.signature(component.__init__).parameters
    if "buttons" in params:  # Gradio 6+
        return {"buttons": ["copy"]}
    if "show_copy_button" in params:  # Gradio 4-5
        return {"show_copy_button": True}
    return {}


def transcribe_for_ui(pipeline: Pipeline, audio_path: str | None, fmt: str) -> tuple[str, str]:
    """Run one file and write the download. Raises ValueError with a message for people."""
    if not audio_path:
        raise ValueError("Upload or record some audio first.")
    try:
        result = pipeline.transcribe(audio_path)
    except GoftarError as exc:
        raise ValueError(str(exc)) from exc
    out_dir = Path(tempfile.mkdtemp(prefix="goftar-ui-"))
    files = write_transcript(result, out_dir, Path(audio_path).stem, [fmt])
    return result.text or "(no speech found)", str(files[0])


def build_ui(pipeline: Pipeline):
    import gradio as gr

    def run(audio_path: str | None, fmt: str):
        try:
            return transcribe_for_ui(pipeline, audio_path, fmt)
        except ValueError as exc:
            raise gr.Error(str(exc)) from exc

    with gr.Blocks(title="Goftar") as demo:
        gr.Markdown(
            "# Goftar - گفتار\nOffline Persian speech-to-text. Nothing leaves this machine."
        )
        audio = gr.Audio(sources=["upload", "microphone"], type="filepath", label="Audio")
        fmt = gr.Dropdown(sorted(FORMATS), value="txt", label="Download format")
        button = gr.Button("Transcribe", variant="primary")
        text = gr.Textbox(label="Text", lines=12, rtl=True, **copy_button_kwargs(gr.Textbox))
        download = gr.File(label="Download")
        button.click(run, inputs=[audio, fmt], outputs=[text, download])
    return demo
