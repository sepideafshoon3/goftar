# Architecture

```
  voice.ogg
      |
      v
 audio/ingest.py      ffmpeg -> 16 kHz mono WAV (optional denoise); temp files auto-cleaned
      |
      v
 engines/             ASREngine.transcribe(path) -> Transcript   (raw text, time-aligned)
      |
      v
 text/normalize.py    Arabic letters, digits, half-space, punctuation   (pure Python, idempotent)
      |
      v
 core/cache.py        SQLite keyed by sha256(audio) + config fingerprint
      |
      v
 export/formats.py    txt | srt | vtt | json
```

## Modules

| Module | Responsibility |
|---|---|
| `core/types.py` | `Segment`, `Transcript`: the only data structures that cross stage boundaries |
| `core/config.py` | typed config, YAML loading, strict validation, cache fingerprint |
| `core/pipeline.py` | orchestration; lazy engine; one lock so a single loaded model serves one job at a time |
| `engines/` | backend interface + registry; built-ins are imported lazily |
| `text/` | Persian normalisation |
| `metrics.py`, `bench.py` | WER/CER and the benchmark harness |
| `interfaces/` | CLI (Typer), REST (FastAPI), UI (Gradio): thin layers over `Pipeline` |
| `doctor.py` | environment diagnostics |

## Design rules

- Interfaces never contain logic; they build a `Config`, call `Pipeline`, render results.
- Heavy imports (faster-whisper, fastapi, gradio) are lazy, so `goftar version` and the test suite stay fast and work without them.
- Config is strict: unknown keys raise. The cache key hashes every setting that can change output, so changing the model or prompt never serves stale text.
- Offline is a first-class setting (`offline: true`), not an accident of having cached models.
