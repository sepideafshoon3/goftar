import json

import pytest

pytest.importorskip("typer")

from typer.testing import CliRunner  # noqa: E402

from goftar import __version__  # noqa: E402
from goftar.interfaces.cli import app  # noqa: E402

runner = CliRunner()


def test_version():
    result = runner.invoke(app, ["version"])
    assert result.exit_code == 0
    assert result.stdout.strip() == __version__


def test_transcribe_writes_requested_formats(tmp_path, wav_file):
    out = tmp_path / "out"
    result = runner.invoke(
        app,
        ["transcribe", str(wav_file), "--engine", "fake", "-o", str(out), "-f", "txt,json,srt",
         "--no-cache"],
    )  # fmt: skip
    assert result.exit_code == 0, result.output
    assert (out / "voice.txt").read_text(encoding="utf-8").startswith("سلام، حال")
    assert json.loads((out / "voice.json").read_text(encoding="utf-8"))["engine"] == "fake"
    assert (out / "voice.srt").exists()


def test_transcribe_folder_writes_next_to_inputs(tmp_path, wav_file):
    result = runner.invoke(app, ["transcribe", str(tmp_path), "--engine", "fake", "--no-cache"])
    assert result.exit_code == 0, result.output
    assert (tmp_path / "voice.txt").exists()


def test_unknown_format_is_a_usage_error(wav_file):
    result = runner.invoke(app, ["transcribe", str(wav_file), "--engine", "fake", "-f", "docx"])
    assert result.exit_code == 2


def test_folder_without_audio(tmp_path):
    result = runner.invoke(app, ["transcribe", str(tmp_path), "--engine", "fake"])
    assert result.exit_code == 1


def test_bad_config_file(tmp_path, wav_file):
    cfg = tmp_path / "bad.yaml"
    cfg.write_text("modle: small\n", encoding="utf-8")
    result = runner.invoke(app, ["transcribe", str(wav_file), "-c", str(cfg)])
    assert result.exit_code == 2


def test_doctor_runs():
    result = runner.invoke(app, ["doctor"])
    assert "python" in result.output
    assert "ffmpeg" in result.output


def test_models_list_empty(tmp_path):
    result = runner.invoke(app, ["models", "list", "--models-dir", str(tmp_path)])
    assert result.exit_code == 0
    assert "no models" in result.output


def test_dataset_workflow(tmp_path, wav_file):
    import shutil

    folder = tmp_path / "ds"
    folder.mkdir()
    audio = folder / "v1.wav"
    shutil.copy(wav_file, audio)

    status = runner.invoke(app, ["dataset", "status", str(folder)])
    assert status.exit_code == 0
    assert "[todo ] v1.wav" in status.output

    prep = runner.invoke(app, ["dataset", "prepare", str(folder), "--engine", "fake"])
    assert prep.exit_code == 0, prep.output
    assert (folder / "v1.draft.txt").exists()
    assert "[review] v1.wav" in runner.invoke(app, ["dataset", "status", str(folder)]).output

    accepted = runner.invoke(app, ["dataset", "accept", str(audio)])
    assert accepted.exit_code == 0, accepted.output
    assert (folder / "v1.txt").exists()
    assert "[ok   ] v1.wav" in runner.invoke(app, ["dataset", "status", str(folder)]).output

    again = runner.invoke(app, ["dataset", "accept", str(audio)])
    assert again.exit_code == 1


def test_dataset_status_empty_folder(tmp_path):
    assert runner.invoke(app, ["dataset", "status", str(tmp_path)]).exit_code == 1


def test_bench_with_details(tmp_path, wav_file):
    import shutil

    folder = tmp_path / "ds"
    folder.mkdir()
    shutil.copy(wav_file, folder / "v1.wav")
    (folder / "v1.txt").write_text("سلام حال شما چطوره", encoding="utf-8")
    cfg = tmp_path / "fake.yaml"
    cfg.write_text("engine: fake\n", encoding="utf-8")
    out = tmp_path / "run.md"

    result = runner.invoke(
        app,
        ["bench", str(folder), "--models", "m1", "-c", str(cfg), "--out", str(out), "--details"],
    )
    assert result.exit_code == 0, result.output
    assert "| m1 |" in out.read_text(encoding="utf-8")
    assert "Most frequent word errors" in (tmp_path / "run.details.md").read_text(encoding="utf-8")
