# 1. Use faster-whisper as the default engine

Status: accepted

## Context

The project needs free, fully local Persian speech recognition that runs on a laptop
(6-core CPU, optional 4 GB NVIDIA GPU).

## Decision

Default to faster-whisper (CTranslate2) with Whisper models.

## Consequences

- Good Persian accuracy at `medium`/`large-v3`; `int8` quantisation makes CPU use practical.
- Built-in VAD reduces hallucinations on silence.
- Whisper output needs Persian-specific clean-up, hence `text/normalize.py`.
- Other backends (whisper.cpp, Vosk, fine-tuned Persian checkpoints converted to CTranslate2)
  stay possible behind the `ASREngine` interface, and the benchmark decides between them.
