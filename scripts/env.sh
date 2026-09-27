# Source me: developer environment for the scripts in this repo (NOT needed by the public
# build/run/test commands in README.md, which take explicit options or KOKORO_* variables).
# - PROJECT_ROOT: this checkout (derived from this file's location).
# - KOKORO_DATA: the data root (model snapshot, frontend data, fixtures, evidence), from the caller's
#   environment, else from the untracked scripts/env.local.sh (e.g. `export KOKORO_DATA=/path`).
# - Build outputs: cargo's default ./target of this checkout (an inherited CARGO_TARGET_DIR is unset).
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
if [[ -z "${KOKORO_DATA:-}" && -f "$PROJECT_ROOT/scripts/env.local.sh" ]]; then
  source "$PROJECT_ROOT/scripts/env.local.sh"
fi
if [[ -z "${KOKORO_DATA:-}" ]]; then
  echo "scripts/env.sh: KOKORO_DATA is not set (export it, or put 'export KOKORO_DATA=/path' in scripts/env.local.sh)" >&2
  return 1 2>/dev/null || exit 1
fi
export KOKORO_DATA
export HF_HOME="$KOKORO_DATA/hf"
export TORCH_HOME="$KOKORO_DATA/torch"
export UV_CACHE_DIR="$KOKORO_DATA/uv-cache"
export TMPDIR="$KOKORO_DATA/tmp"
mkdir -p "$TMPDIR"
unset CARGO_TARGET_DIR
export KOKORO_VENV="$KOKORO_DATA/reference/venv-prod"
export KOKORO_PY="$KOKORO_VENV/bin/python"
