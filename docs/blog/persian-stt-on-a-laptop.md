# Persian speech-to-text on an ordinary laptop: what I measured

*Draft, December 2026. Numbers from [Goftar](https://github.com/sepideafshoon3/goftar) 0.1.0.*

I wanted to turn Persian voice notes and long recordings into text without sending them to
anyone: no cloud service, no API key, nothing leaving the laptop. The laptop is ordinary: an
RTX 2050 with 4 GB of VRAM, a 6-core i5-11400H and 16 GB of RAM. This post is what I learned
choosing a model for it, with the numbers.

## How I measured

- **Main set**: 20 of my own voice notes (about 5 minutes in total) with hand-checked
  transcripts, including English words inside Persian sentences and one file full of numbers.
- **Held-out set**: 8 more clips (about 3 minutes) that I never looked at while tuning prompts,
  text rules or references. Whenever the two sets disagree, I trust the held-out set.
- **Word error rate (WER)**, after normalising both texts the same way: Arabic `ي`/`ك` become
  Persian `ی`/`ک`, digits are unified, punctuation is ignored, and spacing around `ها`, `تر`,
  `ی` and `می` is ignored. Without the last rule, `کتاب‌ها` against `کتابها` counts as an
  error, and Persian WER numbers become mostly a measure of half-space habits.
- **Speed** as RTF: processing time divided by audio length. 0.1 means a 10-minute recording
  takes one minute.

Twenty-eight clips is a small set. One clip can move the held-out average by several points,
and run-to-run noise is about half a point. Read small differences as ties.

## The numbers

| Model | WER main | WER held-out | Speed (RTF) | RAM | Usable subtitles |
|---|---|---|---|---|---|
| Whisper `small` | 52.0%¹ | | 0.05 | 0.9 GB | yes |
| Whisper `medium` | 21.9%¹ | | 0.11 | 1.1 GB | yes |
| **Whisper `large-v3-turbo`** | 19.8% | 26.9% | 0.09 | 1.5 GB | yes |
| `large-v3-turbo`, CPU only | 18.8% | 29.3% | 0.45 | 1.9 GB | yes |
| MohammadGholizadeh turbo Persian fine-tune | 14.3% | **23.0%** | 0.09 | 0.8 GB | no |
| givia turbo Persian fine-tune | **13.6%** | 27.1% | 0.09 | 0.8 GB | no |
| AmirMohseni large-v3 Persian fine-tune | 15.8% | 37.1% | 0.20 | 1.0 GB | yes |

¹ Measured with an earlier, slightly stricter scoring, on the main set only.

All GPU runs use faster-whisper (CTranslate2) with `int8_float16`, beam size 5 and voice
activity detection on.

## What I learned

**1. `large-v3-turbo` is the right default for a 4 GB GPU.** It is as accurate as `medium` or
better, faster, and with 8-bit weights it needs about 1.3 GB of VRAM. `small` is not usable for
Persian: half the words are wrong.

**2. Fine-tunes look much better until you test them on new audio.** On the main set, the
Persian fine-tunes cut the error rate by almost a third (19.8% to 13.6–15.8%). On the held-out
set, one of them is still clearly better (23.0% against 26.9%), one is a tie, and one is much
worse (37.1%). Without a held-out set I would have picked the wrong model.

**3. Accuracy is not the only column.** Both turbo fine-tunes were trained without timestamp
tokens, so they return one segment per 30 seconds. The text is good; the subtitles are
useless. For a voice-note tool that also writes SRT files, that decides it: the default stays
`large-v3-turbo`, and the fine-tunes are an option for plain text.

**4. Prompts overfit.** Whisper's `initial_prompt` lets you hint at vocabulary. A prompt built
from the main set's mistakes gained 2 points there and lost 6 points on the held-out set: it
pulls the text towards its own words even when the speech is about something else. Beam size,
silence thresholds and similar settings changed nothing beyond the noise.

**5. Voice activity detection matters.** Turning it off cost 7 points on the held-out set.
With it on, silence, background noise and a test tone produced no text at all, instead of the
"subtitles by …" lines Whisper likes to invent.

**6. Check your scoring before your model.** The fine-tunes write numbers as words
(`صد و بیست و پنج`), my references use digits (`۱۲۵`). On the file full of numbers that looked
like 84–87% WER. A rule that turns compound number words into digits brought it to 23–26%. It
only converts compound numbers: in speech, `یه` and `دو` are often not numbers at all.

## Things that only showed up on long files

Short clips hid four problems that a 74-minute podcast exposed:

- **Memory grew with the length of the file**: 5 GB of RAM for 74 minutes, so about 9.5 GB
  would be needed for 3 hours. Transcribing in 10-minute pieces, cut at the quietest moment near
  each boundary, keeps it at 1.9 GB whatever the length, and was slightly faster.
- **"Offline" was not quite offline.** About 15 seconds into a run, onnxruntime (used for voice
  activity detection) sent telemetry with a device ID to a Microsoft server. Its own off
  switch did not stop it; an environment variable did. I only found it by tracing every network
  connection during a long run, which I now recommend to anyone who promises privacy.
- **Batch mode overwrote files**: three `voice.ogg` files in different folders became one
  transcript. Now the output mirrors the input folders.
- **Subtitles had lines of up to 144 characters.** Now: two lines of about 42 characters per
  cue, with long segments split into timed cues.

The GPU stayed at or below 76 °C at 80–92% load for the whole podcast, which took about 7
minutes: 10 times faster than real time.

## Try it

```bash
git clone https://github.com/sepideafshoon3/goftar && cd goftar
./scripts/setup.sh --gpu        # or without --gpu on a machine with no NVIDIA card
conda activate goftar
goftar models download large-v3-turbo
goftar transcribe voice.ogg --offline -f txt,srt
goftar bench path/to/your/clips  # measure on your own voice before you believe my numbers
```

## Caveats

- 28 short clips, about 8 minutes of audio. Your speakers, accent, microphone and topic will
  move these numbers.
- The reference transcripts started as model drafts that I corrected by listening. That can
  favour Whisper-like models a little, however carefully the drafts are fixed.
- Not tested: Persian models that are not Whisper-based, and cloud services (on purpose).
