"""Configuration: typed dataclasses, YAML loading, strict validation.

Unknown keys are rejected so typos in a YAML file fail loudly instead of being ignored.
"""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Any

import yaml

from goftar.errors import ConfigError
from goftar.text.spelling import file_digest

DEVICES = {"auto", "cpu", "cuda"}
DIGIT_MODES = {"persian", "latin", "keep"}


@dataclass
class VADConfig:
    enabled: bool = True
    min_silence_ms: int = 500


@dataclass
class AudioConfig:
    sample_rate: int = 16000
    denoise: bool = False


@dataclass
class PostprocessConfig:
    enabled: bool = True
    fix_arabic_chars: bool = True
    remove_diacritics: bool = True
    digits: str = "persian"  # persian | latin | keep
    half_space: bool = True
    punctuation: bool = True
    use_hazm: bool = False
    spelling: bool = False  # fix homophone-letter typos (needs `wordlist` or hazm)
    wordlist: str = ""  # path: one word per line, optional count; "" = hazm's list
    glossary: str = ""  # path to a YAML mapping: wrong spelling -> right spelling

    def __post_init__(self) -> None:
        if self.digits not in DIGIT_MODES:
            raise ConfigError(f"postprocess.digits must be one of {sorted(DIGIT_MODES)}")


@dataclass
class CacheConfig:
    enabled: bool = True
    path: str = ".cache/goftar.sqlite"


@dataclass
class Config:
    engine: str = "faster_whisper"
    model: str = "small"  # model name, or path to a local CTranslate2 model directory
    models_dir: str = "models"
    device: str = "auto"  # auto | cpu | cuda
    compute_type: str = "auto"  # e.g. int8, int8_float16, float16
    cpu_threads: int = 0  # 0 = library default
    language: str = "fa"  # "auto" to detect
    beam_size: int = 5
    initial_prompt: str = ""
    condition_on_previous_text: bool = False
    offline: bool = False  # forbid any network access (sets HF_HUB_OFFLINE=1)
    vad: VADConfig = field(default_factory=VADConfig)
    audio: AudioConfig = field(default_factory=AudioConfig)
    postprocess: PostprocessConfig = field(default_factory=PostprocessConfig)
    cache: CacheConfig = field(default_factory=CacheConfig)

    def __post_init__(self) -> None:
        if self.device not in DEVICES:
            raise ConfigError(f"device must be one of {sorted(DEVICES)}, got {self.device!r}")
        if self.beam_size < 1:
            raise ConfigError("beam_size must be >= 1")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> Config:
        return _build(cls, data, "root")

    def fingerprint(self) -> str:
        """Hash of every setting that can change the transcript (used for cache keys)."""
        ignored = {"cache", "models_dir", "cpu_threads", "offline"}
        payload = {k: v for k, v in self.to_dict().items() if k not in ignored}
        post = payload["postprocess"]
        for key in ("wordlist", "glossary"):
            if post.get(key):
                post[key + "_digest"] = file_digest(post[key])
        blob = json.dumps(payload, sort_keys=True, ensure_ascii=False)
        return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:16]


_SECTIONS: dict[str, type] = {
    "vad": VADConfig,
    "audio": AudioConfig,
    "postprocess": PostprocessConfig,
    "cache": CacheConfig,
}


def _build(cls: type, data: dict[str, Any] | None, where: str) -> Any:
    if data is None:
        data = {}
    if not isinstance(data, dict):
        raise ConfigError(f"'{where}' must be a mapping, got {type(data).__name__}")
    known = {f.name for f in fields(cls)}
    unknown = sorted(set(data) - known)
    if unknown:
        raise ConfigError(f"unknown key(s) {unknown} in '{where}' (allowed: {sorted(known)})")
    kwargs: dict[str, Any] = {}
    for key, value in data.items():
        if cls is Config and key in _SECTIONS:
            value = _build(_SECTIONS[key], value, key)
        kwargs[key] = value
    return cls(**kwargs)


def deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    out = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = deep_merge(out[key], value)
        else:
            out[key] = value
    return out


def _read_yaml(path: Path) -> dict[str, Any]:
    try:
        data = yaml.safe_load(path.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        raise ConfigError(f"invalid YAML in {path}: {exc}") from exc
    if data is None:
        return {}
    if not isinstance(data, dict):
        raise ConfigError(f"{path} must contain a YAML mapping at the top level")
    return data


def load_config(path: str | Path | None = None, overrides: dict[str, Any] | None = None) -> Config:
    """Load config: explicit path > $GOFTAR_CONFIG > ./goftar.yaml > built-in defaults."""
    candidate = path or os.environ.get("GOFTAR_CONFIG")
    data: dict[str, Any] = {}
    if candidate:
        p = Path(candidate).expanduser()
        if not p.is_file():
            raise ConfigError(f"config file not found: {p}")
        data = _read_yaml(p)
    elif Path("goftar.yaml").is_file():
        data = _read_yaml(Path("goftar.yaml"))
    if overrides:
        data = deep_merge(data, overrides)
    return Config.from_dict(data)


def apply_offline(config: Config) -> None:
    """Make HuggingFace libraries refuse network access when config.offline is set."""
    if config.offline:
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"
