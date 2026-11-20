"""Nothing leaves the machine: third-party telemetry is switched off."""

import os
import subprocess
import sys


def test_importing_goftar_disables_onnxruntime_telemetry():
    env = {k: v for k, v in os.environ.items() if k != "ORT_DISABLE_TELEMETRY"}
    out = subprocess.run(
        [sys.executable, "-c", "import goftar, os; print(os.environ['ORT_DISABLE_TELEMETRY'])"],
        env=env, capture_output=True, text=True, check=True,
    )  # fmt: skip
    assert out.stdout.strip() == "1"


def test_user_choice_is_kept():
    env = {**os.environ, "ORT_DISABLE_TELEMETRY": "0"}
    out = subprocess.run(
        [sys.executable, "-c", "import goftar, os; print(os.environ['ORT_DISABLE_TELEMETRY'])"],
        env=env, capture_output=True, text=True, check=True,
    )  # fmt: skip
    assert out.stdout.strip() == "0"


def test_cli_entry_point_imports_the_package_first():
    # The console script imports goftar.interfaces.cli, which runs goftar/__init__ first.
    import goftar.interfaces.cli  # noqa: F401

    assert os.environ.get("ORT_DISABLE_TELEMETRY") is not None
