# Phase 2 results: Persian fine-tunes, decoding settings, long file

- Date: 2026-11-19. Machine: RTX 2050 (4 GB), i5-11400H, 16 GB RAM, Linux.
- Every run: `int8_float16` on CUDA, `beam_size: 5`, VAD on (500 ms) unless the row says otherwise.
- Main set: 20 files (`bench/dataset`). Held-out: 8 clips (`bench/heldout`) that were not used
  to build prompts, rules or references. **The held-out column is the honest one.**
- Scoring v2, plus number words scored as digits (`postprocess.number_words`, PR #14). That
  only changes the fine-tune rows: they write numbers as words, and turbo writes digits.
- Run-to-run noise is about ±0.5 WER points (turbo held-out: 26.5% and 26.9% on two runs). With
  8 held-out clips one file moves the mean by several points, so read small gaps as ties.

## Persian fine-tunes

| Model | Base | Main WER | Main CER | Held-out WER | Held-out CER | RTF | RAM (MB) | Segment timestamps |
|---|---|---|---|---|---|---|---|---|
| `large-v3-turbo` (default) | turbo | 19.8% | 7.0% | 26.9% | 13.1% | 0.09 | 1498 | yes |
| [MohammadGholizadeh/whisper-large-v3-turbo-persian](https://huggingface.co/MohammadGholizadeh/whisper-large-v3-turbo-persian) (MIT) | turbo | 14.3% | 4.5% | **23.0%** | **9.1%** | 0.09 | 807 | **no** |
| [givia/whisper-large-v3-turbo-persian](https://huggingface.co/givia/whisper-large-v3-turbo-persian) (CC BY 4.0) | turbo | **13.6%** | 4.6% | 27.1% | 11.2% | 0.09 | 815 | **no** |
| [AmirMohseni/whisper-large-v3-persian-ct2-int8](https://huggingface.co/AmirMohseni/whisper-large-v3-persian-ct2-int8) (Apache-2.0) | large-v3 | 15.8% | 4.6% | 37.1% | 15.8% | 0.20 | 953 | yes |

- `hamedkhaledi/whisper-large-v3-persian-ct2-int8` has the same `model.bin` as AmirMohseni's
  (identical sha256), so its numbers are those in the AmirMohseni row. That run is also the
  held-out re-test that ADR 3 asked for: 37.1%, clearly worse than turbo.
- The two turbo fine-tunes were converted with CTranslate2 (`int8_float16`, about 780 MB on disk,
  see the README). They run at the same speed as turbo and use about half the RAM.
- Both turbo fine-tunes **return one segment per 30-second window** (median 30.1 s, against
  2.0 s for turbo). They were trained without timestamp tokens. The `.txt` output is fine, but
  SRT/VTT subtitles would be unusable. `word_timestamps=True` does return plausible word timings,
  so the segments can be re-split at pauses and punctuation. This is a follow-up task.
- Their worst main-set file before the number rule was the numbers test (84–87% WER: they
  write `صد و بیست و پنج`, the reference has `۱۲۵`). After the rule: 23–26%.

## Decoding settings (turbo)

| Setting | Main WER | Main CER | Held-out WER | Held-out CER | RTF |
|---|---|---|---|---|---|
| default (beam 5, VAD 500 ms, no prompt) | 19.8% | 7.0% | 26.9% | 13.1% | 0.09 |
| `initial_prompt`: domain words (code, Python, AI, model, dataset, …) | 17.8% | 5.9% | 33.0% | 15.9% | 0.08 |
| `initial_prompt`: a neutral colloquial sentence | 20.9% | 7.9% | 38.2% | 17.4% | 0.08 |
| `beam_size: 1` | 20.0% | 7.2% | 33.7% | 17.3% | 0.07 |
| `beam_size: 8` | 19.2% | 6.5% | 30.2% | 14.8% | 0.11 |
| `vad.min_silence_ms: 250` | 19.6% | 6.9% | 25.3% | 12.4% | 0.09 |
| `vad.min_silence_ms: 1000` | 19.4% | 6.5% | 30.0% | 13.7% | 0.09 |
| `vad.enabled: false` | 19.5% | 6.5% | 34.2% | 13.1% | 0.08 |
| `condition_on_previous_text: true` | 19.8% | 7.0% | 26.0% | 12.4% | 0.08 |

- **`initial_prompt` overfits.** A prompt built from the main set's errors gains 2 points there
  and loses 6 on the held-out clips (worse on 6 of 8). The prompt pulls the decoder toward its
  own words even when the speech is about something else. Keep it empty by default, and use
  one only for recordings that really are about the prompt's topic.
- Beam size, VAD and conditioning: no setting beats the default on both sets by more than the
  noise. `vad.min_silence_ms: 250` is slightly better on both sets (0.2 and 1.6 points), which
  is too little to change the default on 28 clips.
- Turning VAD off costs 7 points on the held-out set: keep it on.

## Long file (74-minute podcast, no reference)

Checked with a script for repeated segments, n-gram loops, text-free gaps, very high or very
low characters per second, and known Whisper hallucinations (`زیرنویس`, `امضا`, subtitle
credits).

| Model | Time | Speed | Peak RAM | Segments | Words | Loops / duplicates | Gaps > 20 s | Hallucinated phrases |
|---|---|---|---|---|---|---|---|---|
| `large-v3-turbo` | 7 m 16 s | 10x real time | 5.0 GB | 2073 | 11,259 | 1 / 2 | 0 | 0 |
| MohammadGholizadeh | 4 m 22 s | 17x | 4.6 GB | 149 | 9,534 | 2 / 0 | 0 | 0 |
| givia | 4 m 13 s | 18x | 4.6 GB | 147 | 9,935 | 0 / 0 | 0 | 0 |

- No hallucination loops with `condition_on_previous_text: false`. Every flagged repeat was a
  real repeat by the speaker ("این نشد بعدی، این نشد بعدی", "آره آره آره آره").
- GPU peaked at 76 °C and 1.3 GB VRAM.
- The fine-tunes write about 13% fewer words than turbo, evenly across the file, so no part of
  the recording is missing. They may drop fillers and repetitions, or turbo may add words. This
  needs a hand-checked 2–3 minute excerpt to settle.
- Peak RAM of 4.6–5 GB for a 74-minute file is high (the decoded audio itself is about
  300 MB). To investigate in phase 3 (batch and stability).
