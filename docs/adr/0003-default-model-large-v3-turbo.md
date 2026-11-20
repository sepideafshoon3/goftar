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

## Held-out check

Same model and settings on 8 held-out clips (`bench/results-heldout.md`):

| Set | Samples | WER (v2) | CER (v2) | RTF |
|---|---|---|---|---|
| Main | 20 | 19.8% | 7.0% | 0.08 |
| Held-out | 8 | 26.5% | 13.8% | 0.10 |

The gap is mostly one file: an 11 s clip scores 74% WER, while the other seven range from 3% to
29%. Colloquial speech and short clips hurt most. With 8 samples this is a rough check, not a
confirmation. The fine-tune above has not yet been re-tested on this set.

## Consequences

- About as accurate as `medium`, but roughly 2x faster and well below real time on the GPU.
- Fits in 4 GB of VRAM with int8_float16.
- The model mixes scripts for English words: sometimes Persian (`پایثان`), sometimes Latin
  (`mlp`). A glossary or transliteration step is the next lever, not a bigger model.
- `small` is not good enough for Persian (52% WER) and stays only as a quick test model.
- Open: the held-out set (8 clips) shows higher error than the main set, and a long (10+ minute)
  file still needs to confirm the numbers. Revisit this ADR if they disagree.

## Revisited 2026-11-19 (phase 2)

Full numbers in [`bench/results-phase2.md`](../../bench/results-phase2.md).

- **Long file:** turbo on a 74-minute podcast runs at 10x real time with no hallucination loops,
  no text-free gaps and no subtitle-credit phrases. The default holds on long recordings.
- **AmirMohseni fine-tune on held-out:** 37.1% WER against 26.9% for turbo. The open question
  above is settled: not adopted.
- **Turbo-based Persian fine-tunes** are better than turbo on text. MohammadGholizadeh scores
  14.3% / 23.0% WER (main / held-out) against 19.8% / 26.9%, at the same speed and half the
  RAM. But they return one segment per 30-second window (no timestamp tokens), so SRT/VTT
  output would be unusable.
- **Decoding settings:** none beats the defaults on both sets. A domain `initial_prompt` helps
  the set it was written for and hurts the held-out set by 6 points, so it stays empty.

Decision: keep `large-v3-turbo` as the default for now. Re-evaluate the switch to
MohammadGholizadeh once its long segments are re-split using word timestamps (which it does
produce) and a short hand-checked excerpt of a long file shows that its ~13% lower word count
is fillers it leaves out, not speech it misses. Users who only need `.txt` can already use it
(see "Persian fine-tunes" in the README).
