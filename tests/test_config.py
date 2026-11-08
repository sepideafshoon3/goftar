from pathlib import Path

import pytest

from goftar import __version__
from goftar.core.config import Config, load_config
from goftar.errors import ConfigError

ROOT = Path(__file__).resolve().parents[1]


def test_default_yaml_matches_code_defaults():
    import yaml

    shipped = yaml.safe_load((ROOT / "config" / "default.yaml").read_text(encoding="utf-8"))
    assert shipped == Config().to_dict()


def test_unknown_key_is_rejected():
    with pytest.raises(ConfigError, match="unknown key"):
        Config.from_dict({"modle": "small"})


def test_unknown_nested_key_is_rejected():
    with pytest.raises(ConfigError, match="vad"):
        Config.from_dict({"vad": {"enabeld": True}})


@pytest.mark.parametrize(
    "data",
    [{"device": "tpu"}, {"beam_size": 0}, {"postprocess": {"digits": "roman"}}],
)
def test_invalid_values_are_rejected(data):
    with pytest.raises(ConfigError):
        Config.from_dict(data)


def test_partial_nested_override_keeps_other_defaults():
    cfg = Config.from_dict({"vad": {"min_silence_ms": 900}})
    assert cfg.vad.min_silence_ms == 900
    assert cfg.vad.enabled is True


def test_load_config_from_file_and_overrides(tmp_path, monkeypatch):
    monkeypatch.delenv("GOFTAR_CONFIG", raising=False)
    path = tmp_path / "c.yaml"
    path.write_text("model: medium\ndevice: cpu\nvad:\n  enabled: false\n", encoding="utf-8")
    cfg = load_config(path, overrides={"device": "cuda", "cache": {"enabled": False}})
    assert cfg.model == "medium"
    assert cfg.device == "cuda"  # override wins over file
    assert cfg.vad.enabled is False
    assert cfg.cache.enabled is False


def test_load_config_missing_file(tmp_path):
    with pytest.raises(ConfigError, match="not found"):
        load_config(tmp_path / "nope.yaml")


def test_env_var_selects_config(tmp_path, monkeypatch):
    path = tmp_path / "env.yaml"
    path.write_text("model: tiny\n", encoding="utf-8")
    monkeypatch.setenv("GOFTAR_CONFIG", str(path))
    assert load_config().model == "tiny"


def test_fingerprint_tracks_output_affecting_settings_only():
    base = Config().fingerprint()
    assert Config(model="medium").fingerprint() != base
    assert Config(initial_prompt="x").fingerprint() != base
    assert Config(cpu_threads=4).fingerprint() == base
    assert Config.from_dict({"cache": {"enabled": False}}).fingerprint() == base


def test_version_matches_pyproject():
    import re

    text = (ROOT / "pyproject.toml").read_text(encoding="utf-8")
    declared = re.search(r'^version = "([^"]+)"', text, re.MULTILINE).group(1)
    assert declared == __version__
