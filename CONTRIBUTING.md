# Contributing

## Workflow

1. `./scripts/setup.sh` once (conda env, git hooks).
2. Branch from an up-to-date `main`: `git switch -c feat/persian-half-space`
3. Commit small, with Conventional Commit messages.
4. `make lint test`, then push and open a pull request.
5. CI must be green. PRs are **squash-merged**: the PR title becomes the commit on `main`.

`main` is protected: direct commits are blocked locally by a hook and on GitHub by branch protection.

## Branch names

`<type>/<short-kebab-description>`

| type | use for |
|---|---|
| `feat` | new capability |
| `fix` | bug fix |
| `docs` | documentation only |
| `refactor` / `perf` | internal change / speed-memory improvement |
| `test` | tests only |
| `build` / `ci` / `chore` | packaging, pipelines, housekeeping |
| `hotfix` | urgent fix |

Also allowed: `main`, `develop`, `release/vX.Y.Z`, `dependabot/*`.
Good: `feat/persian-half-space`, `fix/ogg-duration`, `feat/12-add-vad`. Bad: `my-branch`, `Feat/Upper`, `feat/under_score`.

## Commit messages

```
<type>(<scope>): <subject>

<optional body: why, not what>

<optional footer: Closes #123>
```

- types: `feat fix docs style refactor perf test build ci chore revert`
- scopes (optional, lowercase): `core audio engines text export cli api ui bench ci deps docs`
- header at most 72 characters, imperative mood, no trailing period
- breaking change: `feat(api)!: rename /transcribe`
- a git commit template (`.gitmessage`) is installed by `setup.sh`

## What enforces this

| Where | Check | Bypassable? |
|---|---|---|
| `pre-commit` hook | no commit on `main`, branch name, ruff, whitespace/YAML/large files/private keys | `--no-verify` |
| `commit-msg` hook | Conventional Commits format | `--no-verify` |
| `pre-push` hook | branch name + test suite | `--no-verify` |
| CI `conventions` job | branch name, PR title, every commit in the PR | **no** |
| CI `lint`, `test`, `build` | ruff, pytest on Python 3.10-3.12, package builds | **no** |
| Branch protection on `main` | CI required, linear history, no force-push | no (admins can override) |

Hooks are a convenience for fast feedback; CI and branch protection are the real gate.

## Releasing

1. Bump `version` in `pyproject.toml` and `__version__` in `src/goftar/__init__.py` via a `chore(release): v0.2.0` PR.
2. After it is merged: `git tag v0.2.0 && git push origin v0.2.0`.
3. The `Release` workflow checks the tag matches the version, builds, and publishes a GitHub release with generated notes.

## Adding an ASR engine

Subclass `goftar.engines.base.ASREngine`, implement `transcribe()` (and optionally `load()`), and
register it in `goftar/engines/__init__.py` (`_BUILTIN`). Add a test using the `FakeEngine`
pattern from `tests/conftest.py`, and a row in the benchmark so it can be compared fairly.
