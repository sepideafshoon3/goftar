"""Local web UI (Gradio). Record or upload audio, get Persian text, timed segments and files."""

from __future__ import annotations

import html
import inspect
import re
import tempfile
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from goftar.core.pipeline import Pipeline
from goftar.core.types import Transcript
from goftar.errors import EngineError, GoftarError
from goftar.export.formats import write_transcript

DOWNLOAD_FORMATS = ("txt", "srt", "vtt")
MAX_FILE_SIZE = "500mb"  # a 74-minute m4a is about 70 MB

INTRO = """# گفتار
گفتار فارسی را به متن تبدیل می‌کند.
همه‌چیز روی همین کامپیوتر انجام می‌شود و صدای شما جایی فرستاده نمی‌شود.

صدا را **ضبط** یا فایل را **آپلود** کنید؛ تبدیل خودکار شروع می‌شود.
برای گفتار ساخته شده است، نه آهنگ.
"""

# Mark the document as Persian and right-to-left before Gradio renders it.
HEAD = (
    "<script>document.documentElement.dir = 'rtl'; document.documentElement.lang = 'fa';</script>"
)

# Right-to-left page. The segment table is plain HTML (Gradio's Dataframe draws its rows
# left-to-right in a monospace font); its time columns stay left-to-right.

CSS = """
.gradio-container {
  direction: rtl;
  text-align: right;
  font-family: Vazirmatn, "Noto Sans Arabic", Tahoma, sans-serif;
}
/* In RTL the labels sit on the right, so the clear/copy buttons go to the left. */
.gradio-container [data-testid="block-label"] { direction: rtl; }
.gradio-container .icon-button-wrapper.top-panel { right: auto; left: 0; }
#goftar-intro, #goftar-summary { overflow: visible !important; }
.gradio-container button.label-wrap { direction: rtl; }
#goftar-segments table { width: 100%; border-collapse: collapse; direction: rtl; }
#goftar-segments th, #goftar-segments td {
  text-align: right;
  padding: 6px 10px;
  border-bottom: 1px solid var(--border-color-primary);
  vertical-align: top;
}
#goftar-segments .t { direction: ltr; white-space: nowrap; font-variant-numeric: tabular-nums; }
"""

_FA_DIGITS = str.maketrans("0123456789", "۰۱۲۳۴۵۶۷۸۹")

# AudioError reasons (audio/ingest.py) in Persian. Unknown reasons are shown as they are.
_REASONS_FA = {
    "the file is empty": "فایل خالی است",
    "it has no audio track": "فایل صدا ندارد",
    "it is not an audio file, or it is damaged": "فایل صوتی نیست یا خراب است",
    "the file is incomplete (an unfinished download or recording?)": (
        "فایل ناقص است (دانلود یا ضبطِ نیمه‌کاره؟)"
    ),
    "permission denied": "اجازه‌ی خواندن فایل وجود ندارد",
}
_DECODE_ERROR = re.compile(r"^could not decode (?P<name>.+?): (?P<reason>.+)$", re.S)


def fa_digits(value: object) -> str:
    return str(value).translate(_FA_DIGITS)


def clock(seconds: float) -> str:
    """12.3 -> 00:12, 3725 -> 1:02:05."""
    s = int(seconds)
    h, m, s = s // 3600, s % 3600 // 60, s % 60
    return f"{h}:{m:02d}:{s:02d}" if h else f"{m:02d}:{s:02d}"


def persian_error(exc: Exception) -> str:
    """A message for people, in Persian, for an error from the pipeline."""
    msg = str(exc)
    m = _DECODE_ERROR.match(msg)
    if m:
        reason = _REASONS_FA.get(m["reason"], m["reason"])
        return f"فایل «{m['name']}» خوانده نشد: {reason}."
    if isinstance(exc, EngineError):
        return f"مدل اجرا نشد: {msg}"
    return msg


def copy_button_kwargs(component: Any) -> dict[str, Any]:
    """The copy-button option for this Gradio version (renamed in Gradio 6)."""
    params = inspect.signature(component.__init__).parameters
    if "buttons" in params:  # Gradio 6+
        return {"buttons": ["copy"]}
    if "show_copy_button" in params:  # Gradio 4-5
        return {"show_copy_button": True}
    return {}


def segments_html(rows: list[list[str]]) -> str:
    """The segment table as right-to-left HTML (start, end, text)."""
    if not rows:
        return ""
    body = "".join(
        f'<tr><td class="t">{html.escape(a)}</td><td class="t">{html.escape(b)}</td>'
        f"<td>{html.escape(t)}</td></tr>"
        for a, b, t in rows
    )
    return (
        '<table dir="rtl"><thead><tr><th>شروع</th><th>پایان</th><th>متن</th></tr></thead>'
        f"<tbody>{body}</tbody></table>"
    )


@dataclass
class UIResult:
    text: str
    rows: list[list[str]]  # [start, end, text] per segment
    files: list[str]
    summary: str


def summarize(result: Transcript) -> str:
    if not result.text.strip():
        return "صحبتی در این صدا پیدا نشد."
    parts = [f"مدت صدا: {fa_digits(clock(result.duration))}"]
    if result.processing_time is not None:
        seconds = fa_digits(f"{result.processing_time:.1f}").replace(".", "٫")  # Persian decimal
        parts.append(f"زمان تبدیل: {seconds} ثانیه")
    parts.append(f"{fa_digits(len(result.segments))} بخش")
    return " · ".join(parts)


def transcribe_for_ui(
    pipeline: Pipeline,
    audio_path: str | None,
    progress: Callable[[float], None] | None = None,
) -> UIResult:
    """Run one file and write the downloads. Raises ValueError with a Persian message."""
    if not audio_path:
        raise ValueError("اول صدا را ضبط یا یک فایل صوتی آپلود کنید.")
    try:
        result = pipeline.transcribe(audio_path, progress=progress)
    except GoftarError as exc:
        raise ValueError(persian_error(exc)) from exc
    out_dir = Path(tempfile.mkdtemp(prefix="goftar-ui-"))
    files = write_transcript(result, out_dir, Path(audio_path).stem, DOWNLOAD_FORMATS)
    rows = [[clock(s.start), clock(s.end), s.text] for s in result.segments if s.text.strip()]
    return UIResult(result.text, rows, [str(f) for f in files], summarize(result))


def build_ui(pipeline: Pipeline):
    import gradio as gr

    def run(audio_path: str | None, progress=gr.Progress()):  # noqa: B008 - Gradio's API
        if not audio_path:  # cleared the player: clear the outputs too
            return "", "", None, ""
        progress(0, desc="آماده‌سازی صدا…")
        try:
            r = transcribe_for_ui(pipeline, audio_path, lambda f: progress(f, desc="تبدیل به متن…"))
        except ValueError as exc:
            raise gr.Error(str(exc), print_exception=False) from None
        return r.text, segments_html(r.rows), r.files, r.summary

    with gr.Blocks(title="گفتار") as demo:
        gr.Markdown(INTRO, rtl=True, elem_id="goftar-intro")
        audio = gr.Audio(
            sources=["microphone", "upload"],
            type="filepath",
            label="ضبط یا آپلود صدا",
            elem_id="goftar-audio",
        )
        button = gr.Button("تبدیل به متن", variant="primary")
        summary = gr.Markdown(rtl=True, elem_id="goftar-summary")
        text = gr.Textbox(
            label="متن (قابل ویرایش)",
            lines=10,
            rtl=True,
            interactive=True,
            **copy_button_kwargs(gr.Textbox),
        )
        with gr.Accordion("بخش‌ها با زمان", open=False):
            table = gr.HTML(elem_id="goftar-segments")
        files = gr.File(
            label="دانلود: txt (متن)، srt و vtt (زیرنویس)",
            file_count="multiple",
            interactive=False,
        )
        outputs = [text, table, files, summary]
        button.click(run, inputs=audio, outputs=outputs)
        audio.stop_recording(run, inputs=audio, outputs=outputs)
        audio.upload(run, inputs=audio, outputs=outputs)

    # One transcription at a time: a second one would compete for the same GPU memory.
    demo.queue(default_concurrency_limit=1, max_size=20)
    return demo


def launch_options() -> dict[str, Any]:
    """Options for demo.launch() (Gradio 6 takes css there, not in Blocks)."""
    return {"css": CSS, "head": HEAD, "max_file_size": MAX_FILE_SIZE}
