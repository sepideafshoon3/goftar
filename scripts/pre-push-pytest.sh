#!/usr/bin/env bash
# Pre-push hook: run the fast test suite with the project's Python, even when the
# goftar env is not activated in the shell that runs `git push`.
set -euo pipefail

if python -c "import goftar, pytest" 2>/dev/null; then
  exec python -m pytest -q -x
fi
if command -v conda >/dev/null 2>&1; then
  prefix=$(conda env list | awk '$1 == "goftar" {print $NF}')
  if [ -n "$prefix" ] && [ -x "$prefix/bin/python" ]; then
    exec "$prefix/bin/python" -m pytest -q -x
  fi
fi
echo "pytest not found: run 'conda activate goftar' (or pip install -e '.[dev]') and push again" >&2
exit 1
