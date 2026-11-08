# Goftar (گفتار)

[![CI](https://github.com/niktapournabi/goftar/actions/workflows/ci.yml/badge.svg)](https://github.com/niktapournabi/goftar/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

**Offline, free Persian speech-to-text.** Drop in a voice note or a long recording, get clean
Persian text, subtitles or JSON. Everything runs on your own machine: no cloud, no API keys,
no data leaving the laptop.

- **Pluggable engines**: faster-whisper today; a one-class interface for adding others.
- **Persian-aware post-processing**: Arabic letter variants, digits, half-spaces (ZWNJ), punctuation.
- **Four interfaces** on one core: CLI (batch), REST API (async jobs), web UI, Python library.
- **Measured, not guessed**: built-in benchmark (WER / CER / speed / RAM) on your own audio.
- **Fully offline**: download a model once, then `offline: true` guarantees no network access.
- **Cached**: the same file with the same settings is never transcribed twice.

## Quick start

```bash
git clone https://github.com/niktapournabi/goftar && cd goftar
./scripts/setup.sh            # conda env + hooks (add --gpu for NVIDIA, --no-github to skip GitHub)
conda activate goftar

goftar doctor                 # checks ffmpeg, GPU, models
goftar models download small  # one-time download (needs internet)
goftar transcribe voice.ogg --offline -f txt,srt
```

```bash
goftar transcribe ~/voices -r -o out/ -f txt,json   # whole folder, recursive
goftar serve                                        # REST API on http://127.0.0.1:8000/docs
goftar ui                                           # web UI on http://127.0.0.1:7860
goftar bench bench/dataset --models small,medium    # WER / CER / RTF / RAM table
```

```python
from goftar import Pipeline, Config

result = Pipeline(Config(model="small", device="cpu")).transcribe("voice.ogg")
print(result.text)
```

## Choosing a model

| Hardware | Model | `device` | `compute_type` |
|---|---|---|---|
| CPU only (6 cores) | `small` / `medium` | `cpu` | `int8` |
| NVIDIA, 4 GB VRAM | `medium` | `cuda` | `int8_float16` |
| NVIDIA, 8 GB+ VRAM | `large-v3` | `cuda` | `float16` |

Do not trust the table: run `goftar bench` on your own recordings and decide with numbers.
Results from the maintainer's laptop are published in [`bench/results.md`](bench/results.md).

## Architecture

```
CLI | REST API | Web UI          interfaces/
            |
         Pipeline                core/pipeline.py   (cache, lock, batch)
  ingest -> ASR -> post-process
  ffmpeg   engine   Persian text rules
            |
   ASREngine (abstract)          engines/           faster_whisper | your engine
```

See [docs/architecture.md](docs/architecture.md) and the decision records in [docs/adr](docs/adr).

## Configuration

Copy [`config/default.yaml`](config/default.yaml) to `goftar.yaml` (or use `--config`). Unknown keys
are rejected, so typos fail loudly. CLI flags override the file.

## REST API

| Endpoint | Purpose |
|---|---|
| `GET /health` | status, version, active model |
| `POST /transcribe?format=json\|txt\|srt\|vtt` | upload a file, get the result synchronously |
| `POST /jobs` then `GET /jobs/{id}` | queue long files and poll |

```bash
curl -F "file=@voice.ogg" http://127.0.0.1:8000/transcribe
```

The server binds to `127.0.0.1` by default. There is no authentication: do not expose it
to a network without a reverse proxy that adds it.

## Development

```bash
make lint    # ruff
make test    # pytest (no model download needed: tests use a fake engine)
```

Branches are named `<type>/<kebab-description>` and commits follow
[Conventional Commits](https://www.conventionalcommits.org/); git hooks and CI enforce both.
Details in [CONTRIBUTING.md](CONTRIBUTING.md).

## License

MIT. Whisper model weights are released by OpenAI under the MIT license.
