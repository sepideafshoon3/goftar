# Common problems

Run `goftar doctor` first. It checks ffmpeg, faster-whisper, the GPU and the local models,
and most of the problems below show up there.

## Downloads and proxies

**`Using SOCKS proxy, but the 'socksio' package is not installed`** when downloading a model.
HuggingFace downloads go through `httpx`, which needs `socksio` to talk to a SOCKS proxy. It
is a Goftar dependency, so this only happens if Goftar was not installed with
`pip install -e .`. Fix: `pip install socksio`.

**`Missing dependencies for SOCKS support`** from `pip` while installing. pip needs `PySocks`
for a SOCKS proxy, and it cannot install it through that same proxy. Either install it once
without the proxy (`ALL_PROXY= pip install pysocks`), or point pip at an HTTP proxy instead:
`pip install --proxy http://127.0.0.1:PORT ...`.

**Use `socks5h://`, not `socks5://`,** in `ALL_PROXY` / `HTTPS_PROXY` if names do not resolve
without the proxy. With `socks5h` the proxy does the DNS lookup.

**Model downloads are slow or stop half-way.** Re-run the same `goftar models download`
command: it resumes from where it stopped. `large-v3-turbo` is about 1.6 GB.

**Can I check that nothing is sent anywhere?** Set `offline: true` (or pass `--offline`) after
the model is downloaded. Goftar then sets `HF_HUB_OFFLINE=1`, and it turns off the telemetry of
onnxruntime (used by the voice-activity filter), which otherwise contacts Microsoft during long
runs. To verify on Linux:

```bash
strace -f -e trace=connect -o /tmp/net.log goftar transcribe voice.ogg --offline
grep -E 'AF_INET6?' /tmp/net.log     # should print nothing
```

## Disk space

What takes space, measured on Linux:

| Item | Size | Where |
|---|---|---|
| conda env with everything (`goftar[all]`, ffmpeg) | about 2.5 GB | `~/miniconda3/envs/goftar` |
| GPU libraries (`--gpu`: cuBLAS, cuDNN) | 2.1 GB more | same env |
| `large-v3-turbo` model | 1.6 GB | `models/large-v3-turbo` |
| `medium` / `large-v3` model | 1.5 GB / 3.1 GB | `models/<name>` |
| transcript cache | small (text only) | `.cache/goftar.sqlite` |

**`[Errno 28] No space left on device` during `pip install`** usually means the temporary
folder is full, not the disk you install to. pip builds in `$TMPDIR` (often a small `/tmp`).
Point it somewhere bigger: `TMPDIR=~/tmp-pip pip install -e .[all]`.

**Free space after experimenting:**

```bash
conda clean --all                 # downloaded conda packages
pip cache purge                   # pip's wheel cache
goftar models list                # then delete models/<name> you no longer use
du -sh ~/.cache/huggingface       # models downloaded outside goftar end up here
```

Move models to another disk with `models_dir: /path/to/models` in `goftar.yaml`, and the
HuggingFace cache with `HF_HOME=/path/to/hf`.

## GPU and CUDA

Goftar uses the GPU when one is visible (`device: auto`). It needs an NVIDIA driver for
CUDA 12 (driver 525 or newer). The CUDA toolkit itself is not needed: `./scripts/setup.sh --gpu`
installs the cuBLAS and cuDNN libraries into the env.

**`Library libcublas.so.12 is not found or cannot be loaded`** or
**`Unable to load any of {libcudnn_ops.so.9...}`**. The CUDA libraries are not on the loader
path. Run `./scripts/setup.sh --gpu`, then **deactivate and re-activate the env**
(`conda deactivate && conda activate goftar`): the path is set by the env's activate script.
The same error appears when goftar is started without activating the env, for example as
`~/miniconda3/envs/goftar/bin/goftar serve` from a service, cron job or IDE run button. Start it
with `conda run -n goftar goftar serve` instead, which runs the activate script.
To transcribe right away on the CPU instead: `goftar transcribe voice.ogg --device cpu`.

**`goftar doctor` shows `cuda (ctranslate2): 0 device(s) visible`.** Check `nvidia-smi`. If it
fails, the driver is missing or not loaded (often after a kernel update: reboot). Goftar falls
back to the CPU, which works but is about 5 to 10 times slower.

**`CUDA failed with error out of memory`.** `large-v3-turbo` with `int8_float16` uses about
1.3 GB of VRAM, so it fits on a 4 GB GPU, but not if a browser or a game is using the rest.
Close those, or use `compute_type: int8_float16` if you changed it to `float16`, or use
`--model medium`. Long files do not need more VRAM: audio is processed in 10-minute pieces.

**GPU temperature.** On a laptop RTX 2050, a 74-minute file kept the GPU at 74 to 76 °C at
80 to 92% load. That is normal for a laptop GPU.

## Audio files

**`could not decode voice.xyz: it is not an audio file, or it is damaged`.** ffmpeg could not
read it. Try playing it in a media player. If it plays, convert it: `ffmpeg -i voice.xyz voice.wav`.

**`ffmpeg not found; passing voice.ogg to the engine undecoded`.** Goftar still works for
common formats, but denoising and some formats need ffmpeg. Install it in the env with
`conda install -c conda-forge ffmpeg`, or system-wide (`sudo apt install ffmpeg`).

**The result is `(no speech found)`.** The voice-activity filter found no speech. That is the
right answer for silence or noise. For very quiet speech, try `vad.enabled: false` in
`goftar.yaml` (it can make Whisper invent text in silent parts).

**Repeated or invented sentences.** Keep `condition_on_previous_text: false` (the default) and
leave `initial_prompt` empty unless the recording really is about the prompt's topic. A prompt
pulls the text towards its own words.

## Web UI

**The microphone button does nothing.** Browsers allow the microphone only on `localhost` or
HTTPS. Open `http://127.0.0.1:7860`, not the machine's network address.

**Live text waits until I stop recording.** Use the **Live (زنده)** tab: the microphone in the
**File** tab records first and transcribes when you stop. In the Live tab, text appears about
a second after you speak on a GPU; on a CPU it can fall many seconds behind.

**Live text is greyed out.** Grey is the quick first guess. It is redone with the full
settings, and turns solid, at the next pause in your speech.

**`TypeError: ... unexpected keyword argument 'show_copy_button'`** when starting the UI.
Old Goftar with Gradio 6. Update Goftar (fixed in 0.1.0).
