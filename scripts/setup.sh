#!/usr/bin/env bash
# One-shot setup: conda env + git repo + commit/branch hooks + GitHub repo with protections.
#
#   ./scripts/setup.sh                 # env + hooks + public GitHub repo
#   ./scripts/setup.sh --gpu           # also install CUDA libs for NVIDIA GPUs
#   ./scripts/setup.sh --private       # private GitHub repo (branch protection needs a paid plan)
#   ./scripts/setup.sh --no-github     # local only
set -euo pipefail
cd "$(dirname "$0")/.."

ENV_NAME="goftar"
REPO_NAME="goftar"
VISIBILITY="public"
GPU=0
GITHUB=1

while [[ $# -gt 0 ]]; do
  case "$1" in
    --gpu) GPU=1 ;;
    --private) VISIBILITY="private" ;;
    --no-github) GITHUB=0 ;;
    --env) ENV_NAME="${2:?--env needs a name}"; shift ;;
    --repo) REPO_NAME="${2:?--repo needs a name}"; shift ;;
    -h|--help) sed -n '2,8p' "$0"; exit 0 ;;
    *) echo "unknown option: $1" >&2; exit 2 ;;
  esac
  shift
done

info() { printf '\033[1;34m==>\033[0m %s\n' "$*"; }
warn() { printf '\033[1;33mwarn:\033[0m %s\n' "$*" >&2; }
die()  { printf '\033[1;31merror:\033[0m %s\n' "$*" >&2; exit 1; }

command -v git >/dev/null || die "git is not installed"
CONDA="${CONDA_EXE:-$(command -v conda || true)}"
[[ -n "$CONDA" ]] || die "conda not found. Install Miniconda/Miniforge first."
in_env() { "$CONDA" run --no-capture-output -n "$ENV_NAME" "$@"; }

# 1) conda environment ------------------------------------------------------------------------
if "$CONDA" env list | awk '{print $1}' | grep -qx "$ENV_NAME"; then
  info "updating conda env '$ENV_NAME'"
  "$CONDA" env update -n "$ENV_NAME" -f environment.yml --prune
else
  info "creating conda env '$ENV_NAME' (python 3.11 + ffmpeg + goftar[all])"
  "$CONDA" env create -n "$ENV_NAME" -f environment.yml
fi

if [[ "$GPU" == 1 ]]; then
  info "installing CUDA runtime libraries for faster-whisper"
  in_env python -m pip install -e ".[gpu]"
  PREFIX="$(in_env python -c 'import sys; print(sys.prefix)')"
  mkdir -p "$PREFIX/etc/conda/activate.d"
  cat > "$PREFIX/etc/conda/activate.d/goftar_cuda.sh" <<'EOF'
# Put pip-installed cuBLAS/cuDNN on the loader path whenever the env is activated.
_goftar_libs="$(python - <<'PY' 2>/dev/null
import os
try:
    import nvidia.cublas.lib as a, nvidia.cudnn.lib as b
    print(os.path.dirname(a.__file__) + ":" + os.path.dirname(b.__file__))
except Exception:
    pass
PY
)"
if [ -n "$_goftar_libs" ]; then
  export LD_LIBRARY_PATH="$_goftar_libs${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
fi
unset _goftar_libs
EOF
  info "GPU libs wired into the env's activate hook (re-activate the env to pick them up)"
fi

# 2) git repository -----------------------------------------------------------------------------
if [[ ! -d .git ]]; then
  info "initialising git repository on 'main'"
  git init -b main
fi
git config user.name >/dev/null || die "set your identity first: git config --global user.name 'Your Name'"
git config user.email >/dev/null || die "set your identity first: git config --global user.email you@example.com"

if ! git rev-parse --verify HEAD >/dev/null 2>&1; then
  info "creating the initial commit (hooks are installed afterwards)"
  rm -f condaenv.*.requirements.txt   # stray file written by `conda env create`
  git add -A
  git commit -q -m "chore: initial project scaffold"
fi

# 3) hooks -----------------------------------------------------------------------------------------
info "installing pre-commit, commit-msg and pre-push hooks"
in_env pre-commit install --install-hooks -t pre-commit -t commit-msg -t pre-push
git config commit.template .gitmessage

# 4) GitHub ----------------------------------------------------------------------------------------
if [[ "$GITHUB" == 1 ]]; then
  command -v gh >/dev/null || die "GitHub CLI not found (https://cli.github.com). Re-run with --no-github to skip."
  gh auth status >/dev/null 2>&1 || die "not logged in: run 'gh auth login' (grant the 'workflow' scope), then re-run."

  if ! git remote get-url origin >/dev/null 2>&1; then
    info "creating $VISIBILITY GitHub repository '$REPO_NAME'"
    gh repo create "$REPO_NAME" "--$VISIBILITY" --source=. --remote=origin \
      --description "Offline Persian speech-to-text: pluggable ASR engines, Persian post-processing, CLI, API, UI"
  fi
  # The first push skips local hooks: setup may run outside the conda env, and CI is the real gate.
  info "pushing main"
  git push --no-verify -u origin main

  SLUG="$(gh repo view --json nameWithOwner -q .nameWithOwner)"
  info "configuring $SLUG (squash-merge only, PR title becomes the commit message)"
  gh repo edit "$SLUG" --enable-squash-merge --enable-merge-commit=false \
    --enable-rebase-merge=false --delete-branch-on-merge >/dev/null \
    || warn "could not update merge settings"
  gh api -X PATCH "repos/$SLUG" -f squash_merge_commit_title=PR_TITLE \
    -f squash_merge_commit_message=PR_BODY >/dev/null \
    || warn "could not set squash commit defaults"

  info "protecting 'main' (CI must pass, linear history, no force-push)"
  if ! gh api -X PUT "repos/$SLUG/branches/main/protection" --input - >/dev/null <<'JSON'
{
  "required_status_checks": {
    "strict": true,
    "contexts": ["lint", "conventions", "test (3.10)", "test (3.11)", "test (3.12)", "build"]
  },
  "enforce_admins": false,
  "required_pull_request_reviews": null,
  "restrictions": null,
  "required_linear_history": true,
  "allow_force_pushes": false,
  "allow_deletions": false
}
JSON
  then
    warn "branch protection failed (private repos on the free plan do not support it)"
  fi
fi

cat <<EOF

Done. Next steps:
  conda activate $ENV_NAME
  goftar doctor
  goftar models download small
  git switch -c feat/my-first-change     # branch names are enforced by the hook
EOF
