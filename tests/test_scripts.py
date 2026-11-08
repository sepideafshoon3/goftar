"""The git-hook scripts are tested exactly as git/CI will invoke them: as subprocesses."""

import subprocess
import sys
from pathlib import Path

import pytest

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
COMMIT = SCRIPTS / "check_commit_msg.py"
BRANCH = SCRIPTS / "check_branch_name.py"


def run(script: Path, *args: str) -> subprocess.CompletedProcess:
    return subprocess.run([sys.executable, str(script), *args], capture_output=True, text=True)


@pytest.mark.parametrize(
    "message",
    [
        "feat: add vad options",
        "fix(audio): handle ogg without duration",
        "feat(api)!: rename /transcribe",
        "chore(deps): bump faster-whisper",
        "docs: explain offline mode\n\nLonger body.\n\nCloses #4",
        "Merge branch 'main' into feat/x",
        "fixup! feat: add vad options",
        'Revert "feat: add vad options"',
        "feat: real subject\n# comment lines are ignored by git",
    ],
)
def test_valid_commit_messages(message):
    result = run(COMMIT, "--message", message)
    assert result.returncode == 0, result.stderr


@pytest.mark.parametrize(
    "message",
    [
        "",
        "added stuff",
        "Feat: capitalised type",
        "feat:no space",
        "feat(Core): uppercase scope",
        "feature: wrong type",
        "feat: ends with a period.",
        "feat: " + "x" * 80,
        "feat: ok header\nbody without blank line",
    ],
)
def test_invalid_commit_messages(message):
    result = run(COMMIT, "--message", message)
    assert result.returncode == 1
    assert "Expected format" in result.stderr


def test_commit_message_from_file_like_the_git_hook(tmp_path):
    good = tmp_path / "COMMIT_EDITMSG"
    good.write_text("fix(text): keep ZWNJ before suffixes\n", encoding="utf-8")
    assert run(COMMIT, str(good)).returncode == 0
    bad = tmp_path / "BAD_MSG"
    bad.write_text("oops\n", encoding="utf-8")
    assert run(COMMIT, str(bad)).returncode == 1


@pytest.mark.parametrize(
    "name",
    [
        "main",
        "develop",
        "feat/persian-half-space",
        "fix/ogg-duration",
        "docs/readme",
        "ci/cache-pip",
        "chore/bump-ruff",
        "feat/12-add-vad",
        "release/v1.2.0",
        "dependabot/pip/ruff-0.9",
        "hotfix/crash-on-empty-audio",
    ],
)
def test_valid_branch_names(name):
    assert run(BRANCH, "--name", name).returncode == 0


@pytest.mark.parametrize(
    "name",
    ["my-branch", "Feat/Upper", "feat/", "feat/two words", "feat/under_score", "wip", "feat//x"],
)
def test_invalid_branch_names(name):
    result = run(BRANCH, "--name", name)
    assert result.returncode == 1
    assert "<type>/<short-kebab-description>" in result.stderr


def test_detached_head_is_not_blocked():
    assert run(BRANCH, "--name", "HEAD").returncode == 0


def test_commit_range_mode(tmp_path):
    def git(*args):
        subprocess.run(["git", *args], cwd=tmp_path, check=True, capture_output=True)

    git("init", "-b", "main")
    git("config", "user.name", "t")
    git("config", "user.email", "t@example.com")
    (tmp_path / "f").write_text("1")
    git("add", "f")
    git("commit", "-m", "chore: base")
    (tmp_path / "f").write_text("2")
    git("commit", "-am", "feat: good one")
    (tmp_path / "f").write_text("3")
    git("commit", "-am", "bad message")

    def check(rev_range):
        return subprocess.run(
            [sys.executable, str(COMMIT), "--range", rev_range],
            cwd=tmp_path, capture_output=True, text=True,
        )  # fmt: skip

    assert check("HEAD~2..HEAD~1").returncode == 0
    bad = check("HEAD~2..HEAD")
    assert bad.returncode == 1
    assert "bad message" in bad.stderr
