# Benchmark results

- Machine: Linux-7.0.0-43-generic-x86_64-with-glibc2.43 / x86_64
- Samples: 8 (hand-checked reference transcripts)
- WER/CER (scoring v2) after Persian normalisation: punctuation ignored, spacing around ها/تر/ی/می affixes ignored
- RTF = processing time / audio duration (lower is faster; < 1 is faster than real time)

| Model | Device | Compute | WER | CER | RTF | Load (s) | RAM (MB) |
|---|---|---|---|---|---|---|---|
| large-v3-turbo | cuda | int8_float16 | 26.5% | 13.8% | 0.10 | 6.9 | 1506 |
