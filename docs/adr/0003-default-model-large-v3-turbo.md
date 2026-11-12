# 3. Default model: large-v3-turbo on GPU (int8_float16)

Status: accepted

## Context

ADR 1 picked faster-whisper but left the model size to the benchmark. Target hardware is a
laptop with a 4 GB NVIDIA GPU (RTX 2050) and a 6-core CPU. We benchmarked on 20 of our own
Persian recordings with hand-checked references (`bench/dataset`, `goftar bench`).

| Model | WER (v1) | CER (v1) | RTF | RAM (MB) |
|---|---|---|---|---|
| small | 52.0% | 18.4% | 0.05 | 890 |
| medium | 21.9% | 7.5% | 0.11 | 1114 |
| large-v3-turbo | 21.2% | 8.2% | 0.06 | 1974 |

With scoring v2 (affix spacing ignored), large-v3-turbo scores **19.6% WER / 7.9% CER**.
After fixing clear typos in the references and writing English words in Persian script, as
`bench/dataset/README.md` requires, it scores **19.5% WER / 7.0% CER**. That is the current
baseline in `bench/results.md` (a re-run gave 19.8% WER / 7.0% CER).

## Decision

Default to `large-v3-turbo`, `device: cuda`, `compute_type: int8_float16`. Users without a
GPU can override this in `goftar.yaml` or with `--device cpu --compute-type int8`.

## Why not a bigger model

The `--details` shared-error analysis showed that about half of the remaining errors are
made by every model: spelling variants, word spacing, English words, and typos in the
references themselves. A larger model cannot fix those. They belong to the text stage
(normalisation, glossary, optional spelling) and to cleaner references.

## Alternative evaluated: Persian fine-tune

`AmirMohseni/whisper-large-v3-persian-ct2-int8` (a Persian fine-tune of large-v3, CTranslate2
int8) on the same 20 files with scoring v2 (`bench/results-amirmohseni.md`):

| Model | WER (v2) | CER (v2) | RTF | RAM (MB) |
|---|---|---|---|---|
| large-v3-turbo | 19.5% | 7.0% | 0.08 | 1504 |
| AmirMohseni fine-tune | 18.2% | 6.4% | 0.20 | 954 |

It wins on 16 of 20 files but breaks down on one (67.7% WER vs 22.6% for turbo), which looks
like a dropped or hallucinated segment. Not adopted for now: the gain is about 1.3 points on a
small set, it is about 2.5x slower, and the one failure is the kind that hurts most on long
recordings. Re-test it against turbo on the held-out set before changing the default.

## Consequences

- About as accurate as `medium`, but roughly 2x faster and well below real time on the GPU.
- Fits in 4 GB of VRAM with int8_float16.
- The model mixes scripts for English words: sometimes Persian (`پایثان`), sometimes Latin
  (`mlp`). A glossary or transliteration step is the next lever, not a bigger model.
- `small` is not good enough for Persian (52% WER) and stays only as a quick test model.
- Open: these numbers come from 20 short clips by a few speakers. A held-out set and a long
  (10+ minute) file still need to confirm them. Revisit this ADR if they disagree.
