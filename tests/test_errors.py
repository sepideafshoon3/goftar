"""Bad input and broken setup: clear one-line errors, no tracebacks, no wasted work."""

import pytest

from goftar.audio.ingest import explain_ffmpeg_error, ffmpeg_available
from goftar.core.pipeline import Pipeline
from goftar.engines import ASREngine, register_engine
from goftar.errors import AudioError, EngineError

needs_ffmpeg = pytest.mark.skipif(not ffmpeg_available(), reason="ffmpeg not installed")


class BrokenEngine(ASREngine):
    """Fails like a missing model would."""

    name = "broken"
    calls = 0

    def transcribe(self, audio_path):
        type(self).calls += 1
        raise EngineError("model 'small' not found")


@pytest.fixture
def broken_config(fake_config):
    register_engine("broken", BrokenEngine)
    BrokenEngine.calls = 0
    fake_config.engine = "broken"
    return fake_config


@pytest.mark.parametrize(
    ("stderr", "expected"),
    [
        ("[in#0 @ 0x5be2] Error opening input: Invalid data found when processing input",
         "not an audio file"),
        ("Error opening input files: End of file", "not an audio file"),
        ("[out#0/wav @ 0x5f35] Output file does not contain any stream", "no audio track"),
        ("[mov,mp4 @ 0x55] moov atom not found", "incomplete"),
        ("[in#0 @ 0x5be2] Something new went wrong\nlast line here", "last line here"),
        ("", "unknown ffmpeg error"),
    ],
)  # fmt: skip
def test_ffmpeg_errors_become_one_short_line(tmp_path, stderr, expected):
    f = tmp_path / "x.ogg"
    f.write_bytes(b"x")
    message = explain_ffmpeg_error(f, stderr)
    assert expected in message
    assert "\n" not in message and "0x" not in message


def test_empty_file_is_named_as_such(tmp_path):
    f = tmp_path / "empty.mp3"
    f.touch()
    assert explain_ffmpeg_error(f, "Invalid data found") == "the file is empty"


@needs_ffmpeg
def test_undecodable_file_error_is_short(fake_config, tmp_path):
    bad = tmp_path / "garbage.ogg"
    bad.write_bytes(b"not audio at all" * 100)
    with pytest.raises(AudioError) as info:
        Pipeline(fake_config).transcribe(bad)
    assert (
        str(info.value) == "could not decode garbage.ogg: it is not an audio file, or it is damaged"
    )


@needs_ffmpeg
def test_engine_error_stops_the_batch(broken_config, wav_file, tmp_path):
    second = tmp_path / "second.wav"
    second.write_bytes(wav_file.read_bytes())
    with pytest.raises(EngineError):
        list(Pipeline(broken_config).transcribe_many([wav_file, second]))
    assert BrokenEngine.calls == 1


@needs_ffmpeg
def test_bad_file_in_batch_logs_no_traceback(fake_config, wav_file, tmp_path, caplog):
    bad = tmp_path / "garbage.mp3"
    bad.write_bytes(b"nope")
    outcomes = dict(Pipeline(fake_config).transcribe_many([bad, wav_file]))
    assert isinstance(outcomes[bad], AudioError)
    assert not [r for r in caplog.records if r.levelname in ("WARNING", "ERROR")]
