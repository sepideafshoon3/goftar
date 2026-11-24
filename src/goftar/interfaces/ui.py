"""Local web UI (Gradio). Record or upload audio, get Persian text, timed segments and files."""

from __future__ import annotations

import contextlib
import html
import inspect
import queue
import re
import tempfile
import threading
from collections.abc import Callable, Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from goftar.core.config import Config
from goftar.core.pipeline import Pipeline
from goftar.core.types import Segment, Transcript
from goftar.errors import EngineError, GoftarError
from goftar.export.formats import write_transcript
from goftar.live import LiveTranscriber

DOWNLOAD_FORMATS = ("txt", "srt", "vtt")
MAX_FILE_SIZE = "500mb"  # a 74-minute m4a is about 70 MB
WEAK_CONFIDENCE = 0.6  # segments below this are marked "check this" in the table

# Mark the document as Persian and right-to-left before Gradio renders it.
HEAD = (
    "<script>document.documentElement.dir = 'rtl'; document.documentElement.lang = 'fa';</script>"
)

# Only local fonts: the page must not fetch anything from the internet (offline promise).
FONT_STACK = ("Vazirmatn", "Noto Sans Arabic", "Noto Naskh Arabic", "Tahoma", "sans-serif")

# Right-to-left page. The segment table is plain HTML (Gradio's Dataframe draws its rows
# left-to-right in a monospace font); its time columns stay left-to-right.
CSS = """
.gradio-container {
  direction: rtl;
  text-align: right;
  font-family: Vazirmatn, "Noto Sans Arabic", "Noto Naskh Arabic", Tahoma, sans-serif;
  max-width: 1180px !important;
  margin: 0 auto !important;
}
:root {
  --g-accent: #7c3aed;
  --g-accent-2: #06b6d4;
  --g-grad: linear-gradient(135deg, #7c3aed 0%, #4f46e5 45%, #06b6d4 100%);
  --g-soft: color-mix(in srgb, var(--g-accent) 9%, transparent);
  --g-warn: #d97706;
}
/* In RTL the labels sit on the right, so the clear/copy buttons go to the left. */
.gradio-container [data-testid="block-label"] { direction: rtl; }
.gradio-container .icon-button-wrapper.top-panel { right: auto; left: 0; }
.gradio-container button.label-wrap { direction: rtl; }
footer { display: none !important; }

/* Header */
#g-hero { overflow: visible !important; }
.g-hero {
  position: relative; overflow: hidden;
  border-radius: 22px; padding: 26px 28px;
  background: var(--g-grad); color: #fff;
  box-shadow: 0 18px 40px -18px rgba(79, 70, 229, .55);
}
.g-hero::after {
  content: ""; position: absolute; inset: auto -60px -90px auto;
  width: 260px; height: 260px; border-radius: 50%;
  background: radial-gradient(circle, rgba(255,255,255,.22), transparent 70%);
}
.g-brand { display: flex; align-items: center; gap: 14px; }
.g-logo {
  width: 52px; height: 52px; border-radius: 16px; flex: none;
  display: grid; place-items: center;
  background: rgba(255,255,255,.16); backdrop-filter: blur(6px);
  border: 1px solid rgba(255,255,255,.28);
}
.g-logo span {
  display: inline-block; width: 4px; margin: 0 1.5px; border-radius: 3px; background: #fff;
  animation: g-wave 1.2s ease-in-out infinite;
}
.g-logo span:nth-child(1) { height: 12px; animation-delay: -.9s; }
.g-logo span:nth-child(2) { height: 22px; animation-delay: -.6s; }
.g-logo span:nth-child(3) { height: 30px; animation-delay: -.3s; }
.g-logo span:nth-child(4) { height: 18px; }
.g-logo span:nth-child(5) { height: 10px; animation-delay: -.45s; }
@keyframes g-wave { 50% { transform: scaleY(.45); } }
@media (prefers-reduced-motion: reduce) { .g-logo span { animation: none; } }
.g-title { font-size: 30px; font-weight: 800; line-height: 1.1; margin: 0; color: #fff; }
.g-sub { margin: 4px 0 0; opacity: .9; font-size: 15px; color: #fff; }
.g-chips { display: flex; flex-wrap: wrap; gap: 8px; margin-top: 18px;
  position: relative; z-index: 1; }
.g-chip {
  font-size: 13px; padding: 5px 12px; border-radius: 999px; color: #fff;
  background: rgba(255,255,255,.15); border: 1px solid rgba(255,255,255,.25);
}
.g-chip b { font-weight: 700; direction: ltr; unicode-bidi: isolate; }

/* Cards */
.g-card {
  border-radius: 20px !important;
  border: 1px solid var(--border-color-primary) !important;
  background: var(--block-background-fill) !important;
  box-shadow: 0 10px 30px -22px rgba(15, 23, 42, .45);
  padding: 18px !important;
}
.g-h { font-size: 16px; font-weight: 700; margin: 0 0 2px;
  display: flex; gap: 8px; align-items: center; }
.g-h .n {
  width: 24px; height: 24px; border-radius: 8px; display: grid; place-items: center;
  font-size: 13px; color: #fff; background: var(--g-grad);
}
#g-run {
  background: var(--g-grad) !important; color: #fff !important; border: 0 !important;
  font-size: 17px !important; font-weight: 700 !important; min-height: 52px;
  border-radius: 14px !important; transition: transform .15s, box-shadow .15s;
  box-shadow: 0 10px 24px -12px rgba(124, 58, 237, .8);
}
#g-run:hover { transform: translateY(-1px); box-shadow: 0 14px 28px -12px rgba(124, 58, 237, .9); }
.g-tips { margin: 4px 0 0; padding: 0 18px 0 0; font-size: 13.5px; line-height: 1.9;
  color: var(--body-text-color-subdued); }
.g-tips li::marker { color: var(--g-accent); }

/* Stats */
#g-stats { overflow: visible !important; }
.g-stats { display: grid; grid-template-columns: repeat(4, minmax(0, 1fr)); gap: 10px; }
.g-stat {
  border-radius: 16px; padding: 12px 14px; background: var(--g-soft);
  border: 1px solid color-mix(in srgb, var(--g-accent) 18%, transparent);
}
.g-stat .k { font-size: 12.5px; color: var(--body-text-color-subdued); }
.g-stat .v { font-size: 22px; font-weight: 800; margin-top: 2px;
  font-variant-numeric: tabular-nums; }
.g-stat .v small { font-size: 13px; font-weight: 500; opacity: .75; }
.g-empty {
  border: 1.5px dashed var(--border-color-primary); border-radius: 16px;
  padding: 22px; text-align: center; color: var(--body-text-color-subdued); font-size: 14.5px;
}
.g-empty .big { font-size: 30px; display: block; margin-bottom: 6px; }
.g-note { margin-top: 10px; font-size: 13px; color: var(--g-warn); }
@media (max-width: 720px) { .g-stats { grid-template-columns: repeat(2, minmax(0, 1fr)); } }

/* Segment table */
#goftar-segments table { width: 100%; border-collapse: separate; border-spacing: 0 6px;
  direction: rtl; }
#goftar-segments th {
  text-align: right; font-size: 12.5px; font-weight: 600; padding: 2px 12px;
  color: var(--body-text-color-subdued);
}
#goftar-segments td { text-align: right; padding: 10px 12px; vertical-align: top; line-height: 1.9;
  background: var(--background-fill-secondary); }
#goftar-segments td:first-child { border-radius: 0 12px 12px 0; }
#goftar-segments td:last-child { border-radius: 12px 0 0 12px; }
#goftar-segments tr[data-start] { cursor: pointer; transition: transform .12s; }
#goftar-segments tr[data-start]:hover td { background: var(--g-soft); }
#goftar-segments tr.on td { background: color-mix(in srgb, var(--g-accent) 16%, transparent); }
#goftar-segments tr.weak td:last-child { box-shadow: inset 3px 0 0 var(--g-warn); }
#goftar-segments .t { direction: ltr; white-space: nowrap; font-variant-numeric: tabular-nums;
  color: var(--g-accent); font-weight: 600; width: 1%; }
#goftar-segments .flag { font-size: 11.5px; color: var(--g-warn); margin-inline-start: 6px;
  white-space: nowrap; }
.g-hint { font-size: 12.5px; color: var(--body-text-color-subdued); margin: 0 0 4px; }
/* Progress while a file is being transcribed */
.g-prog { display: grid; gap: 8px; }
.g-prog .bar { height: 10px; border-radius: 999px; background: var(--g-soft); overflow: hidden; }
.g-prog .bar i { display: block; height: 100%; border-radius: inherit; background: var(--g-grad);
  transition: width .4s ease; }
.g-prog .lbl { display: flex; justify-content: space-between; font-size: 13.5px;
  color: var(--body-text-color-subdued); font-variant-numeric: tabular-nums; }

/* Live tab */
.g-live {
  min-height: 220px; max-height: 460px; overflow-y: auto; padding: 18px 20px;
  border-radius: 16px; background: var(--background-fill-secondary);
  font-size: 19px; line-height: 2.1; direction: rtl; text-align: right;
}
.g-live .tent { color: var(--body-text-color-subdued); }
.g-live .caret { display: inline-block; width: 2px; height: 1.1em; vertical-align: -0.2em;
  margin-inline-start: 3px; background: var(--g-accent); animation: g-blink 1s steps(1) infinite; }
@keyframes g-blink { 50% { opacity: 0; } }
.g-live .idle { color: var(--body-text-color-subdued); font-size: 15px; }
.g-live-meta { display: flex; gap: 8px; align-items: center; margin-bottom: 8px; font-size: 13px;
  color: var(--body-text-color-subdued); }
.g-dot { width: 9px; height: 9px; border-radius: 50%; background: #ef4444;
  box-shadow: 0 0 0 0 rgba(239, 68, 68, .6); animation: g-pulse 1.4s infinite; }
@keyframes g-pulse { 70% { box-shadow: 0 0 0 9px rgba(239, 68, 68, 0); } }
@media (prefers-reduced-motion: reduce) { .g-live .caret, .g-dot { animation: none; } }
.g-foot { text-align: center; font-size: 12.5px; color: var(--body-text-color-subdued);
  padding: 8px 0 18px; }
"""

# Clicking a segment row asks the server to move the player to its start.
SEGMENTS_JS = """
element.addEventListener('click', (e) => {
  const tr = e.target.closest('tr[data-start]');
  if (!tr) return;
  element.querySelectorAll('tr.on').forEach((r) => r.classList.remove('on'));
  tr.classList.add('on');
  trigger('select', { index: Number(tr.dataset.i), value: Number(tr.dataset.start) });
});
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


def fa_decimal(value: float, places: int = 1) -> str:
    """3.42 -> ۳٫۴ (Persian digits and decimal separator)."""
    return fa_digits(f"{value:.{places}f}").replace(".", "٫")


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


def segments_html(
    rows: list[list[str]],
    starts: list[float] | None = None,
    weak: list[bool] | None = None,
) -> str:
    """The segment table as right-to-left HTML (start, end, text).

    With `starts`, each row carries its start time so a click can move the player there.
    With `weak`, low-confidence rows are marked for checking.
    """
    if not rows:
        return ""
    body = []
    for i, (a, b, t) in enumerate(rows):
        attrs = ""
        if starts is not None:
            attrs = f' data-i="{i}" data-start="{starts[i]:.2f}"'
        is_weak = bool(weak and weak[i])
        if is_weak:
            attrs += ' class="weak"'
        flag = '<span class="flag">بررسی شود</span>' if is_weak else ""
        body.append(
            f'<tr{attrs}><td class="t">{html.escape(a)}</td><td class="t">{html.escape(b)}</td>'
            f"<td>{html.escape(t)}{flag}</td></tr>"
        )
    return (
        '<table dir="rtl"><thead><tr><th>شروع</th><th>پایان</th><th>متن</th></tr></thead>'
        f"<tbody>{''.join(body)}</tbody></table>"
    )


@dataclass
class UIResult:
    text: str
    rows: list[list[str]]  # [start, end, text] per segment
    files: list[str]
    summary: str
    stats: str = ""  # stat cards (HTML)
    starts: list[float] = field(default_factory=list)  # start second per row
    weak: list[bool] = field(default_factory=list)  # low confidence per row


def summarize(result: Transcript) -> str:
    if not result.text.strip():
        return "صحبتی در این صدا پیدا نشد."
    parts = [f"مدت صدا: {fa_digits(clock(result.duration))}"]
    if result.processing_time is not None:
        parts.append(f"زمان تبدیل: {fa_decimal(result.processing_time)} ثانیه")
    parts.append(f"{fa_digits(len(result.segments))} بخش")
    return " · ".join(parts)


def empty_stats(message: str = "هنوز چیزی تبدیل نشده است.") -> str:
    return f'<div class="g-empty"><span class="big">🎙️</span>{html.escape(message)}</div>'


def stats_html(result: Transcript, rows: int, weak: int) -> str:
    """Four stat cards: duration, processing time, speed, words."""
    if not result.text.strip():
        return empty_stats("صحبتی در این صدا پیدا نشد.")
    duration = result.duration or 0.0
    took = result.processing_time
    cards = [("مدت صدا", fa_digits(clock(duration)), "")]
    if took is not None:
        cards.append(("زمان تبدیل", fa_decimal(took), "ثانیه"))
        if took > 0 and duration > 0:
            cards.append(("سرعت", f"{fa_decimal(duration / took)}×", "زمان واقعی"))
    words = len(result.text.split())
    cards.append(("کلمه", fa_digits(words), f"در {fa_digits(rows)} بخش"))
    tiles = "".join(
        f'<div class="g-stat"><div class="k">{k}</div><div class="v">{v} <small>{u}</small></div>'
        "</div>"
        for k, v, u in cards
    )
    note = ""
    if weak:
        note = (
            f'<div class="g-note">⚠ {fa_digits(weak)} بخش با اطمینان پایین؛ '
            "در تب «بخش‌ها» علامت خورده‌اند.</div>"
        )
    return f'<div class="g-stats">{tiles}</div>{note}'


def transcribe_for_ui(
    pipeline: Pipeline,
    audio_path: str | None,
    progress: Callable[[float], None] | None = None,
    on_segment: Callable[[Segment], None] | None = None,
) -> UIResult:
    """Run one file and write the downloads. Raises ValueError with a Persian message."""
    if not audio_path:
        raise ValueError("اول صدا را ضبط یا یک فایل صوتی آپلود کنید.")
    try:
        if on_segment is None:
            result = pipeline.transcribe(audio_path, progress=progress)
        else:
            result = pipeline.transcribe(audio_path, progress=progress, on_segment=on_segment)
    except GoftarError as exc:
        raise ValueError(persian_error(exc)) from exc
    out_dir = Path(tempfile.mkdtemp(prefix="goftar-ui-"))
    files = write_transcript(result, out_dir, Path(audio_path).stem, DOWNLOAD_FORMATS)
    kept = [s for s in result.segments if s.text.strip()]
    rows = [[clock(s.start), clock(s.end), s.text] for s in kept]
    weak = [s.confidence is not None and s.confidence < WEAK_CONFIDENCE for s in kept]
    return UIResult(
        result.text,
        rows,
        [str(f) for f in files],
        summarize(result),
        stats=stats_html(result, len(rows), sum(weak)),
        starts=[s.start for s in kept],
        weak=weak,
    )


@dataclass
class Partial:
    """What is known so far while a file is still being transcribed."""

    segments: list[Segment]
    fraction: float  # 0.0 to 1.0


def stream_for_ui(pipeline: Pipeline, audio_path: str | None) -> Iterator[Partial | UIResult]:
    """Transcribe in a worker thread; yield a Partial whenever there is news, then the UIResult.

    Errors from transcribe_for_ui (ValueError with a Persian message) are raised here.
    """
    events: queue.Queue[tuple[str, Any]] = queue.Queue()
    box: dict[str, Any] = {}

    def work() -> None:
        try:
            box["result"] = transcribe_for_ui(
                pipeline,
                audio_path,
                progress=lambda f: events.put(("progress", f)),
                on_segment=lambda s: events.put(("segment", s)),
            )
        except BaseException as exc:  # handed to the caller's thread below
            box["error"] = exc
        finally:
            events.put(("done", None))

    threading.Thread(target=work, name="goftar-ui-transcribe", daemon=True).start()
    segments: list[Segment] = []
    fraction = 0.0
    while True:
        batch = [events.get()]
        with contextlib.suppress(queue.Empty):
            while True:  # take everything that is already waiting: one screen update for all
                batch.append(events.get_nowait())
        done = False
        for kind, value in batch:
            if kind == "segment":
                segments.append(value)
            elif kind == "progress":
                fraction = max(fraction, float(value))
            else:
                done = True
        if done:
            break
        yield Partial(list(segments), fraction)
    if "error" in box:
        raise box["error"]
    yield box["result"]


def progress_html(fraction: float, segments: int) -> str:
    pct = max(0, min(100, round(fraction * 100)))
    label = "آماده‌سازی صدا…" if pct == 0 and not segments else "در حال تبدیل به متن…"
    found = f"{fa_digits(segments)} بخش تا این‌جا" if segments else ""
    return (
        f'<div class="g-prog"><div class="lbl"><span>{label}</span>'
        f"<span>{found} {fa_digits(pct)}٪</span></div>"
        f'<div class="bar"><i style="width:{pct}%"></i></div></div>'
    )


def live_html(committed: str, tentative: str = "", listening: bool = False) -> str:
    """The live text: committed text, then the tentative part greyed out, then a caret."""
    if not committed and not tentative:
        idle = (
            "گوش می‌دهم… شروع کنید به صحبت."
            if listening
            else ("دکمه‌ی ضبط را بزنید و صحبت کنید؛ متن همان‌طور که حرف می‌زنید این‌جا نوشته می‌شود.")
        )
        body = f'<span class="idle">{idle}</span>'
    else:
        body = html.escape(committed)
        if tentative:
            body += f' <span class="tent">{html.escape(tentative)}</span>'
    caret = '<span class="caret"></span>' if listening else ""
    meta = (
        '<div class="g-live-meta"><span class="g-dot"></span>در حال شنیدن</div>'
        if listening
        else ""
    )
    return f'{meta}<div class="g-live" dir="rtl">{body}{caret}</div>'


def _device_label(config: Config) -> str:
    try:
        from goftar.engines.faster_whisper_engine import resolve_device

        device, _ = resolve_device(config.device, config.compute_type)
    except Exception:  # no ctranslate2, or a broken CUDA: just show the setting
        device = config.device
    return {"cuda": "GPU", "cpu": "CPU"}.get(device, device)


def hero_html(config: Config) -> str:
    model = html.escape(Path(config.model).name)
    network = "بدون اینترنت" if config.offline else "روی همین دستگاه"
    return f"""
<div class="g-hero">
  <div class="g-brand">
    <div class="g-logo" aria-hidden="true"><div>
      <span></span><span></span><span></span><span></span><span></span>
    </div></div>
    <div>
      <h1 class="g-title">گفتار</h1>
      <p class="g-sub">گفتار فارسی را به متن، زیرنویس و فایل تبدیل کنید؛
      بی‌آنکه صدایتان جایی برود.</p>
    </div>
  </div>
  <div class="g-chips">
    <span class="g-chip">🔒 {network}</span>
    <span class="g-chip">مدل: <b>{model}</b></span>
    <span class="g-chip">پردازش: <b>{_device_label(config)}</b></span>
    <span class="g-chip">خروجی: <b>txt · srt · vtt</b></span>
  </div>
</div>
"""


LIVE_TIPS = """
<ul class="g-tips">
  <li>متن خاکستری حدس اولیه است؛ با هر مکث، دوباره و دقیق‌تر نوشته می‌شود.</li>
  <li>دقت حالت زنده کمی پایین‌تر از تبدیل فایل است. برای متن نهایی یک جلسه‌ی مهم،
  ضبط کامل را در تب «فایل» تبدیل کنید.</li>
</ul>
"""

TIPS = """
<ul class="g-tips">
  <li>با <b>ضبط</b> یا <b>آپلود</b>، تبدیل خودکار شروع می‌شود.</li>
  <li>برای گفتار ساخته شده است، نه آهنگ.</li>
  <li>متن هم‌زمان با تبدیل، بخش‌به‌بخش ظاهر می‌شود.</li>
  <li>روی هر بخش در تب «بخش‌ها» بزنید تا صدا از همان‌جا پخش شود.</li>
</ul>
"""


def build_ui(pipeline: Pipeline):
    import gradio as gr

    def run(audio_path: str | None):
        if not audio_path:  # cleared the player: clear the outputs too
            yield "", "", None, empty_stats()
            return
        yield "", "", None, progress_html(0, 0)
        try:
            for item in stream_for_ui(pipeline, audio_path):
                if isinstance(item, Partial):
                    segs = item.segments
                    rows = [[clock(x.start), clock(x.end), x.text] for x in segs]
                    yield (
                        " ".join(x.text for x in segs),
                        segments_html(rows),
                        None,
                        progress_html(item.fraction, len(segs)),
                    )
                else:
                    r = item
                    yield r.text, segments_html(r.rows, r.starts, r.weak), r.files, r.stats
        except ValueError as exc:
            raise gr.Error(str(exc), print_exception=False) from None

    def live_decode(samples, fast: bool) -> str:
        try:
            return " ".join(x.text for x in pipeline.transcribe_samples(samples, fast=fast))
        except GoftarError as exc:
            raise gr.Error(persian_error(exc), print_exception=False) from None

    def live_start():
        return LiveTranscriber(live_decode), live_html("", listening=True), ""

    def live_feed(chunk, session):
        if session is None:
            session = LiveTranscriber(live_decode)
        if chunk is not None:
            sample_rate, data = chunk
            session.feed(sample_rate, data)
        return live_html(session.text, session.tentative, listening=True), session

    def live_stop(session):
        if session is None:
            return live_html(""), "", None
        text = session.finish()
        return live_html(text), text, None

    def seek(evt: gr.SelectData):
        try:
            return gr.Audio(playback_position=float(evt.value))
        except (TypeError, ValueError):
            return gr.Audio()

    # Gradio finds SelectData by annotation; with postponed annotations it is only a string.
    seek.__annotations__["evt"] = gr.SelectData

    with gr.Blocks(title="گفتار") as demo:
        gr.HTML(hero_html(pipeline.config), elem_id="g-hero")
        with gr.Tabs(elem_id="g-modes"):
            with gr.Tab("📁 فایل"), gr.Row(equal_height=False):
                with gr.Column(scale=5, min_width=320, elem_classes="g-card"):
                    gr.HTML('<div class="g-h"><span class="n">۱</span>صدا</div>')
                    audio = gr.Audio(
                        sources=["microphone", "upload"],
                        type="filepath",
                        label="ضبط یا آپلود صدا",
                        elem_id="goftar-audio",
                    )
                    button = gr.Button("✨ تبدیل به متن", variant="primary", elem_id="g-run")
                    gr.HTML(TIPS)
                with gr.Column(scale=7, min_width=360, elem_classes="g-card"):
                    gr.HTML('<div class="g-h"><span class="n">۲</span>نتیجه</div>')
                    stats = gr.HTML(empty_stats(), elem_id="g-stats")
                    with gr.Tabs():
                        with gr.Tab("📝 متن"):
                            text = gr.Textbox(
                                label="متن (قابل ویرایش)",
                                lines=12,
                                rtl=True,
                                interactive=True,
                                placeholder="متن اینجا ظاهر می‌شود…",
                                **copy_button_kwargs(gr.Textbox),
                            )
                        with gr.Tab("⏱️ بخش‌ها"):
                            gr.HTML('<p class="g-hint">روی هر بخش بزنید تا صدا به آن‌جا برود.</p>')
                            table = gr.HTML(elem_id="goftar-segments", js_on_load=SEGMENTS_JS)
                        with gr.Tab("⬇️ دانلود"):
                            files = gr.File(
                                label="txt (متن)، srt و vtt (زیرنویس)",
                                file_count="multiple",
                                interactive=False,
                            )
            with gr.Tab("🔴 زنده"):
                with gr.Row(equal_height=False):
                    with gr.Column(scale=4, min_width=300, elem_classes="g-card"):
                        gr.HTML('<div class="g-h"><span class="n">۱</span>میکروفون</div>')
                        live_mic = gr.Audio(
                            sources=["microphone"],
                            type="numpy",
                            streaming=True,
                            label="ضبط زنده",
                            elem_id="g-live-mic",
                        )
                        gr.HTML(LIVE_TIPS)
                    with gr.Column(scale=8, min_width=360, elem_classes="g-card"):
                        gr.HTML('<div class="g-h"><span class="n">۲</span>متن زنده</div>')
                        live_box = gr.HTML(live_html(""), elem_id="g-live-box")
                        live_text = gr.Textbox(
                            label="متن نهایی (بعد از توقف ضبط، قابل ویرایش)",
                            lines=6,
                            rtl=True,
                            interactive=True,
                            **copy_button_kwargs(gr.Textbox),
                        )
                live_session = gr.State(None)
        gr.HTML(
            '<div class="g-foot">گفتار · متن‌باز و آفلاین · '
            "هیچ داده‌ای از این دستگاه خارج نمی‌شود</div>"
        )

        outputs = [text, table, files, stats]
        # The progress bar is drawn in the stats card, so Gradio's overlay is not needed;
        # it would also hide the text that streams in.
        for event in (button.click, audio.stop_recording, audio.upload):
            event(run, inputs=audio, outputs=outputs, show_progress="hidden")
        table.select(seek, outputs=audio, queue=False)

        live_mic.start_recording(
            live_start, outputs=[live_session, live_box, live_text], queue=False
        )
        live_mic.stream(
            live_feed,
            inputs=[live_mic, live_session],
            outputs=[live_box, live_session],
            stream_every=0.5,
            time_limit=3600,
            concurrency_limit=1,
            show_progress="hidden",
        )
        live_mic.stop_recording(
            live_stop, inputs=live_session, outputs=[live_box, live_text, live_session]
        )

    # One transcription at a time: a second one would compete for the same GPU memory.
    demo.queue(default_concurrency_limit=1, max_size=20)
    return demo


def theme():
    """A violet theme with local fonts only (Gradio's defaults may load web fonts)."""
    import gradio as gr

    return gr.themes.Soft(
        primary_hue="violet",
        secondary_hue="cyan",
        neutral_hue="slate",
        radius_size="lg",
        font=FONT_STACK,
        font_mono=("ui-monospace", "Consolas", "monospace"),
    )


def launch_options() -> dict[str, Any]:
    """Options for demo.launch() (Gradio 6 takes css and theme there, not in Blocks)."""
    opts: dict[str, Any] = {"css": CSS, "head": HEAD, "max_file_size": MAX_FILE_SIZE}
    with contextlib.suppress(ImportError):  # no gradio: build_ui fails first anyway
        opts["theme"] = theme()
    return opts
