import pytest

from goftar.audio.ingest import (
    build_ffmpeg_cmd,
    collect_audio_files,
    ffmpeg_available,
    prepared_audio,
)
from goftar.core.config import AudioConfig, Config
from goftar.core.pipeline import Pipeline
from goftar.engines import available_engines, get_engine
from goftar.errors import AudioError, EngineError
from tests.conftest import FakeEngine


def test_pipeline_normalises_engine_output(fake_config, wav_file):
    result = Pipeline(fake_config).transcribe(wav_file)
    assert [s.text for s in result.segments] == [
        "سلام، حال شما چطوره؟",
        "من ۳ تا کتاب‌ها می‌خرم",
    ]
    assert result.engine == "fake"
    assert result.processing_time is not None


def test_second_call_is_served_from_cache(fake_config, wav_file):
    pipe = Pipeline(fake_config)
    first = pipe.transcribe(wav_file)
    second = pipe.transcribe(wav_file)
    assert FakeEngine.calls == 1
    assert second.text == first.text


def test_cache_key_depends_on_settings(fake_config, wav_file):
    Pipeline(fake_config).transcribe(wav_file)
    changed = Config.from_dict({**fake_config.to_dict(), "initial_prompt": "کلمات تخصصی"})
    Pipeline(changed).transcribe(wav_file)
    assert FakeEngine.calls == 2


def test_cache_can_be_disabled(fake_config, wav_file):
    fake_config.cache.enabled = False
    pipe = Pipeline(fake_config)
    pipe.transcribe(wav_file)
    pipe.transcribe(wav_file)
    assert FakeEngine.calls == 2


def test_missing_file(fake_config, tmp_path):
    with pytest.raises(FileNotFoundError):
        Pipeline(fake_config).transcribe(tmp_path / "missing.wav")


@pytest.mark.skipif(not ffmpeg_available(), reason="ffmpeg not installed")
def test_undecodable_file_raises_audio_error(fake_config, tmp_path):
    bad = tmp_path / "garbage.mp3"
    bad.write_bytes(b"this is not audio")
    with pytest.raises(AudioError, match="could not decode"):
        Pipeline(fake_config).transcribe(bad)


@pytest.mark.skipif(not ffmpeg_available(), reason="ffmpeg not installed")
def test_batch_continues_after_a_failure(fake_config, wav_file, tmp_path):
    bad = tmp_path / "garbage.mp3"
    bad.write_bytes(b"nope")
    outcomes = dict(Pipeline(fake_config).transcribe_many([bad, wav_file]))
    assert isinstance(outcomes[bad], Exception)
    assert outcomes[wav_file].text


@pytest.mark.skipif(not ffmpeg_available(), reason="ffmpeg not installed")
def test_prepared_audio_yields_16k_mono_wav_and_cleans_up(wav_file):
    with prepared_audio(wav_file, AudioConfig()) as prepared:
        path = prepared
        assert path.exists()
        assert path.suffix == ".wav"
    assert not path.exists()


def test_ffmpeg_command_options(tmp_path):
    cmd = build_ffmpeg_cmd(tmp_path / "a.ogg", tmp_path / "b.wav", AudioConfig(denoise=True))
    assert cmd[cmd.index("-ar") + 1] == "16000"
    assert cmd[cmd.index("-ac") + 1] == "1"
    assert "-af" in cmd
    assert "-af" not in build_ffmpeg_cmd(tmp_path / "a.ogg", tmp_path / "b.wav", AudioConfig())


def test_collect_audio_files(tmp_path):
    (tmp_path / "a.ogg").write_bytes(b"x")
    (tmp_path / "b.txt").write_bytes(b"x")
    (tmp_path / "sub").mkdir()
    (tmp_path / "sub" / "c.MP3").write_bytes(b"x")
    flat = collect_audio_files([tmp_path])
    assert [p.name for p in flat] == ["a.ogg"]
    deep = collect_audio_files([tmp_path], recursive=True)
    assert sorted(p.name for p in deep) == ["a.ogg", "c.MP3"]


def test_engine_registry():
    assert {"faster_whisper", "fake"} <= set(available_engines())
    with pytest.raises(EngineError, match="unknown engine"):
        get_engine(Config(engine="nope"))


def test_faster_whisper_engine_offline_without_model_is_a_clear_error(tmp_path, monkeypatch):
    from goftar.engines.faster_whisper_engine import FasterWhisperEngine

    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    engine = FasterWhisperEngine(Config(model="small", models_dir=str(tmp_path)))
    with pytest.raises(EngineError, match="goftar models download small"):
        engine.resolve_model()


def test_faster_whisper_engine_prefers_local_model_dir(tmp_path):
    from goftar.engines.faster_whisper_engine import FasterWhisperEngine

    (tmp_path / "small").mkdir()
    engine = FasterWhisperEngine(Config(model="small", models_dir=str(tmp_path)))
    assert engine.resolve_model() == str(tmp_path / "small")
