# Benchmark results

- Machine: Linux-7.0.0-43-generic-x86_64-with-glibc2.43 / x86_64
- Samples: 20 (hand-checked reference transcripts)
- WER/CER (scoring v2) after Persian normalisation: punctuation ignored, spacing around ها/تر/ی/می affixes ignored
- RTF = processing time / audio duration (lower is faster; < 1 is faster than real time)

| Model | Device | Compute | WER | CER | RTF | Load (s) | RAM (MB) |
|---|---|---|---|---|---|---|---|
| AmirMohseni/whisper-large-v3-persian-ct2-int8 | cuda | int8_float16 | 18.2% | 6.4% | 0.20 | 3.1 | 954 |
