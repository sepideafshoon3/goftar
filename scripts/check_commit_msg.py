#!/usr/bin/env python3
"""Validate commit messages against Conventional Commits (stdlib only).

Usage:
    check_commit_msg.py <message-file>        # git commit-msg hook
    check_commit_msg.py --message "feat: x"   # CI: pull-request title
    check_commit_msg.py --range BASE..HEAD    # CI: every commit in a pull request
"""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
from pathlib import Path

TYPES = (
    "feat",
    "fix",
    "docs",
    "style",
    "refactor",
    "perf",
    "test",
    "build",
    "ci",
    "chore",
    "revert",
)
HEADER_RE = re.compile(
    rf"^(?P<type>{'|'.join(TYPES)})"
    r"(?:\((?P<scope>[a-z0-9][a-z0-9._/-]*)\))?"
    r"(?P<breaking>!)?: (?P<subject>\S.*)$"
)
MAX_HEADER = 72
AUTO_PREFIXES = ("Merge ", 'Revert "', "fixup! ", "squash! ", "amend! ")

HELP = f"""
Expected format:  <type>(<scope>): <subject>
  types : {", ".join(TYPES)}
  scope : optional, lowercase (core, audio, engines, text, export, cli, api, ui,
          bench, ci, deps, docs)
  rules : header <= {MAX_HEADER} chars, imperative mood, no trailing period,
          blank line between header and body
Examples:
  feat(text): add half-space fixer for Persian verbs
  fix(audio): handle ogg files with no duration metadata
  feat(api)!: rename /transcribe to /v1/transcribe
"""


def validate(message: str) -> list[str]:
    """Return a list of problems (empty list = valid)."""
    # git strips lines starting with '#', so ignore them too
    lines = [ln for ln in message.splitlines() if not ln.startswith("#")]
    while lines and not lines[0].strip():
        lines.pop(0)
    if not lines:
        return ["commit message is empty"]

    header = lines[0].rstrip()
    if header.startswith(AUTO_PREFIXES):
        return []

    errors: list[str] = []
    match = HEADER_RE.match(header)
    if not match:
        errors.append(f"header does not match '<type>(<scope>): <subject>': {header!r}")
    elif match.group("subject").endswith("."):
        errors.append("subject must not end with a period")
    if len(header) > MAX_HEADER:
        errors.append(f"header is {len(header)} chars; the limit is {MAX_HEADER}")
    if len(lines) > 1 and lines[1].strip():
        errors.append("leave a blank line between the header and the body")
    return errors


def _messages_from_range(rev_range: str) -> list[str]:
    out = subprocess.run(
        ["git", "log", "--format=%B%x1e", rev_range],
        capture_output=True, text=True, check=True,
    ).stdout  # fmt: skip
    return [m.strip("\n") for m in out.split("\x1e") if m.strip()]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument("file", nargs="?", help="commit message file (commit-msg hook)")
    parser.add_argument("--message", help="validate this text directly")
    parser.add_argument("--range", dest="rev_range", help="validate all commits in a git range")
    args = parser.parse_args(argv)

    if args.message is not None:
        messages = [args.message]
    elif args.rev_range:
        messages = _messages_from_range(args.rev_range)
    elif args.file:
        messages = [Path(args.file).read_text(encoding="utf-8")]
    else:
        parser.error("give a message file, --message or --range")

    failed = False
    for message in messages:
        errors = validate(message)
        if errors:
            failed = True
            first_line = next((ln for ln in message.splitlines() if ln.strip()), "")
            print(f"\ninvalid commit message: {first_line!r}", file=sys.stderr)
            for err in errors:
                print(f"  - {err}", file=sys.stderr)
    if failed:
        print(HELP, file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
