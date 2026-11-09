# Benchmark dataset

Put 10-20 of your own Persian recordings here, each with a hand-checked reference transcript:

```
voice01.ogg
voice01.txt      <- exactly what is said (this is what the benchmark reads)
voice02.mp3
voice02.txt
```

Mix clean speech, noisy voice notes, different speakers, some English words mixed in, and one
long file (10+ minutes). Audio files in this folder are git-ignored (they are personal data);
keep only the aggregated numbers in `../results.md`.

## Faster way to get references

Typing transcripts from scratch is slow. Let a model write a draft, then fix it:

```bash
goftar dataset prepare bench/dataset --model medium --device cuda --compute-type int8_float16
# -> voice01.draft.txt (one segment per line)

# open each draft, LISTEN to the audio, and correct every word
goftar dataset status bench/dataset            # ok / review / todo per file
goftar dataset accept bench/dataset/voice01.ogg bench/dataset/voice02.mp3
# -> voice01.draft.txt becomes voice01.txt
```

Drafts are named `*.draft.txt` so an unchecked draft can never be mistaken for a reference.

**Bias warning.** A draft comes from a model, so a reference you only skimmed makes that model
(and similar ones) look better than it is. Draft with your strongest model, fix every word,
and be extra careful with words the model got "plausibly" wrong.

## Run the benchmark

```bash
goftar bench bench/dataset --models small,medium --device cuda --compute-type int8_float16
```
