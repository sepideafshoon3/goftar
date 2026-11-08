# 2. Conventional Commits, short-lived branches, squash merges

Status: accepted

## Context

A tidy history makes releases, changelogs and reviews easier, and the repository is also a
portfolio piece.

## Decision

- Branches: `<type>/<kebab-description>`, short-lived, always from `main`.
- Commits and PR titles: Conventional Commits.
- Merge strategy: squash only; the PR title becomes the commit on `main`. Linear history.
- Enforcement: pre-commit framework hooks locally (fast feedback) and a CI `conventions` job
  plus branch protection (cannot be bypassed with `--no-verify`).
- The checks are stdlib-only scripts in `scripts/`, shared by hooks and CI, and unit-tested.

## Consequences

- No Node toolchain (commitlint) is needed.
- Hooks can be skipped locally, so CI is the source of truth.
