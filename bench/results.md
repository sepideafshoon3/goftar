# Benchmark results

- Machine: Linux-7.0.0-43-generic-x86_64-with-glibc2.43 / x86_64
- Samples: 20 (hand-checked reference transcripts)
- WER/CER computed after Persian normalisation (punctuation and half-space ignored)
- RTF = processing time / audio duration (lower is faster; < 1 is faster than real time)

| Model | Device | Compute | WER | CER | RTF | Load (s) | RAM (MB) |
|---|---|---|---|---|---|---|---|
| small | cuda | int8_float16 | 52.0% | 18.4% | 0.05 | 0.9 | 890 |
| medium | cuda | int8_float16 | 21.9% | 7.5% | 0.11 | 2.0 | 1114 |
| large-v3-turbo | cuda | int8_float16 | 21.2% | 8.2% | 0.06 | 2.0 | 1974 |
