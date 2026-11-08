from goftar.bench import discover_samples, render_markdown, run_benchmark
from goftar.core.config import Config


def test_discover_samples_needs_reference(tmp_path, wav_file):
    (tmp_path / "voice.txt").write_text("سلام حال شما چطوره\n", encoding="utf-8")
    (tmp_path / "orphan.wav").write_bytes(b"x")  # no .txt -> skipped
    samples = discover_samples(tmp_path)
    assert [s.audio.name for s in samples] == ["voice.wav"]
    assert samples[0].reference == "سلام حال شما چطوره"


def test_run_benchmark_and_report(tmp_path, wav_file):
    (tmp_path / "voice.txt").write_text(
        "سلام حال شما چطوره من ۳ تا کتاب ها می خرم", encoding="utf-8"
    )
    samples = discover_samples(tmp_path)
    results = run_benchmark(samples, {"fake-model": Config(engine="fake")})
    assert results[0].error is None
    assert results[0].mean_wer == 0.0
    report = render_markdown(results, n_samples=len(samples))
    assert "| fake-model |" in report
    assert "0.0%" in report


def test_benchmark_reports_engine_errors_instead_of_crashing(tmp_path, wav_file):
    (tmp_path / "voice.txt").write_text("x", encoding="utf-8")
    results = run_benchmark(discover_samples(tmp_path), {"bad": Config(engine="nope")})
    assert results[0].error and "unknown engine" in results[0].error
    assert "error:" in render_markdown(results, n_samples=1)
