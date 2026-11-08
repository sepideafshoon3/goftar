#!/usr/bin/env python3
"""Enforce the branch naming convention (stdlib only).

    <type>/<short-kebab-description>      e.g. feat/persian-half-space, fix/ogg-duration
    main | develop | release/vX.Y.Z | dependabot/*

Usage:
    check_branch_name.py              # uses the current git branch (pre-commit / pre-push hook)
    check_branch_name.py --name NAME  # CI
"""

from __future__ import annotations

import argparse
import re
import subprocess
import sys

BRANCH_TYPES = ("feat", "fix", "docs", "refactor", "perf", "test", "build", "ci", "chore", "hotfix")
PATTERNS = [
    re.compile(r"^(main|develop)$"),
    re.compile(rf"^({'|'.join(BRANCH_TYPES)})/[a-z0-9]+(-[a-z0-9]+)*$"),
    re.compile(r"^release/v?\d+\.\d+\.\d+$"),
    re.compile(r"^dependabot/.+$"),
]

HELP = f"""
Branch names must look like:  <type>/<short-kebab-description>
  types   : {", ".join(BRANCH_TYPES)}
  examples: feat/persian-half-space   fix/ogg-duration   docs/readme-benchmarks   ci/cache-pip
  also ok : main, develop, release/v1.2.0, dependabot/*
Rename with:  git branch -m <new-name>
"""


def is_valid(name: str) -> bool:
    return any(p.match(name) for p in PATTERNS)


def current_branch() -> str:
    result = subprocess.run(
        ["git", "rev-parse", "--abbrev-ref", "HEAD"], capture_output=True, text=True, check=False
    )
    return result.stdout.strip()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument("--name", help="branch name to check (default: current branch)")
    args = parser.parse_args(argv)

    name = args.name if args.name is not None else current_branch()
    # No branch yet, or detached HEAD (rebase, bisect, checking out a tag): nothing to enforce.
    if not name or name == "HEAD":
        return 0
    if is_valid(name):
        return 0
    print(f"invalid branch name: {name!r}", file=sys.stderr)
    print(HELP, file=sys.stderr)
    return 1


if __name__ == "__main__":
    sys.exit(main())
