"""Local web UI (Gradio). Upload or record audio, get RTL text and a downloadable file."""

from __future__ import annotations

import tempfile
from pathlib import Path

from goftar.core.pipeline import Pipeline
from goftar.export.formats import FORMATS, write_transcript


def build_ui(pipeline: Pipeline):
    import gradio as gr

    def run(audio_path: str | None, fmt: str):
        if not audio_path:
            raise gr.Error("Upload or record some audio first.")
        result = pipeline.transcribe(audio_path)
        out_dir = Path(tempfile.mkdtemp(prefix="goftar-ui-"))
        files = write_transcript(result, out_dir, Path(audio_path).stem, [fmt])
        return result.text, str(files[0])

    with gr.Blocks(title="Goftar") as demo:
        gr.Markdown(
            "# Goftar - گفتار\nOffline Persian speech-to-text. Nothing leaves this machine."
        )
        audio = gr.Audio(sources=["upload", "microphone"], type="filepath", label="Audio")
        fmt = gr.Dropdown(sorted(FORMATS), value="txt", label="Download format")
        button = gr.Button("Transcribe", variant="primary")
        text = gr.Textbox(label="Text", lines=12, rtl=True, show_copy_button=True)
        download = gr.File(label="Download")
        button.click(run, inputs=[audio, fmt], outputs=[text, download])
    return demo
