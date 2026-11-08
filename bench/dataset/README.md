# Benchmark dataset

Put 10-20 of your own Persian recordings here, each with a hand-typed reference transcript:

```
voice01.ogg
voice01.txt      <- exactly what is said, typed by you
voice02.mp3
voice02.txt
```

Mix clean speech, noisy voice notes, different speakers and a couple of long files.
Audio files in this folder are git-ignored (they are personal data); keep only the
aggregated numbers in `../results.md`.

Run: `goftar bench bench/dataset --models small,medium`
