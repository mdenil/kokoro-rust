# Source me: canonical project environment. Overrides anything inherited at launch.
# Data root mirrors the project path with /home -> /data (owner convention).
PROJECT_ROOT=/home/mdenil/code/kokoro-rust
export KOKORO_DATA=/data/mdenil/code/kokoro-rust
export HF_HOME="$KOKORO_DATA/hf"
export TORCH_HOME="$KOKORO_DATA/torch"
export UV_CACHE_DIR="$KOKORO_DATA/uv-cache"
export TMPDIR="$KOKORO_DATA/tmp"
export CARGO_TARGET_DIR="$PROJECT_ROOT/target"
export KOKORO_VENV="$KOKORO_DATA/reference/venv-prod"
export KOKORO_PY="$KOKORO_VENV/bin/python"
