# Changelog

Notable changes. Versions follow [Semantic Versioning](https://semver.org/); every change
reaches `main` as a squash-merged pull request with a Conventional Commit title.

## Unreleased

### Added
- Web UI redesign: header showing the model, device and offline status; audio and result cards;
  stat cards (duration, processing time, speed, words); text / segments / downloads tabs.
- Segments appear while a file is still being transcribed, with a progress bar.
- Live tab (experimental): text is written from the microphone while you speak, then
  corrected at each pause.
- Clicking a segment plays the audio from its start; low-confidence segments are marked.
- `ASREngine.transcribe_samples()` and an `on_segment` callback for engines that can stream.

## 0.1.0 - 2026-11-20

First release.

- CLI (single files and recursive batches), REST API with async jobs, Gradio web UI and Python
  library, all on one `Pipeline`.
- faster-whisper engine with `large-v3-turbo` as the default (ADR 3), automatic GPU/CPU choice,
  and long audio in 10-minute pieces so memory stays flat.
- Persian post-processing: letter variants, digits, half-spaces, number words to digits,
  punctuation; optional spelling stage and glossary.
- SRT/VTT subtitles with at most two lines of about 42 characters; txt and json output.
- Benchmark harness (WER, CER, speed, RAM) with a held-out set and a dataset draft workflow.
- Offline mode with onnxruntime telemetry turned off; SQLite cache of results.
- CPU and GPU Docker images.

Full list of pull requests: [v0.1.0 release](https://github.com/sepideafshoon3/goftar/releases/tag/v0.1.0).
