#!/usr/bin/env bash
# Export the spaCy en_core_web_sm 3.8.0 data the native frontend needs into
# DATA_DIR/frontend/spacy-en_core_web_sm-3.8.0/ and verify the result against pinned SHA-256s.
#
#   scripts/prepare_spacy_assets.sh DATA_DIR
#
# This is the only setup step that uses Python (CPython 3.12 with the standard venv module). It
# creates an isolated virtual environment in DATA_DIR/tools/spacy-export (pinned packages from PyPI, the
# model wheel from explosion/spacy-models on GitHub) and runs scripts/export_spacy_assets.py.
# The synthesizer itself never runs Python.
set -euo pipefail

if [[ $# -ne 1 ]]; then
  echo "usage: $0 DATA_DIR" >&2
  exit 2
fi
DATA=$1
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
PY=${PYTHON:-python3.12}
command -v "$PY" >/dev/null || { echo "$0: $PY not found (set PYTHON=/path/to/python3.12)" >&2; exit 2; }
"$PY" -c 'import sys, venv, ensurepip; assert sys.version_info[:2] == (3, 12), sys.version' \
  || { echo "$0: $PY must be CPython 3.12 with the venv and ensurepip modules" >&2; exit 2; }

OUT="$DATA/frontend/spacy-en_core_web_sm-3.8.0"
EXPECTED="b44b0bc0ddd7f742271bb37b4e494953499081cae3e19f14844aab544f9173b9  tokenizer.json
a9162bebd730179e56b6e85a39c5c71b1e925022ffe7e084e6958da8ce893169  lookups.json
3f7694e00ac35ef85e883ff54e52894f6cba6787e1d0499bd349a3e593bc18fb  symbols.json
dfddb78e16608d07741b0a9ee0d000337bbeda3c1732c02e72afe793609dcabb  base_norms.json
f80547c28eb8178bb4f4ab1928b90ebe7f1ca192d8a23877430c357ac55c07f6  model_structure.json
66a133a471e669bedc905373801464665971a562cbea20db347bec56e5e1e43c  tagger_weights.safetensors"

if [[ -d "$OUT" ]] && (cd "$OUT" && echo "$EXPECTED" | sha256sum --check --quiet --status); then
  echo "ok (present)  $OUT"
  exit 0
fi

VENV="$DATA/tools/spacy-export"
MODEL_URL=https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-3.8.0/en_core_web_sm-3.8.0-py3-none-any.whl
MODEL_SHA=1932429db727d4bff3deed6b34cfc05df17794f4a52eeb26cf8928f7c1a0fb85
mkdir -p "$DATA/downloads"
MODEL_WHL="$DATA/downloads/en_core_web_sm-3.8.0-py3-none-any.whl"
if ! [[ -f "$MODEL_WHL" && "$(sha256sum "$MODEL_WHL" | cut -d' ' -f1)" == "$MODEL_SHA" ]]; then
  curl -fL --retry 3 --silent --show-error -o "$MODEL_WHL.part" "$MODEL_URL"
  [[ "$(sha256sum "$MODEL_WHL.part" | cut -d' ' -f1)" == "$MODEL_SHA" ]] || { echo "$0: checksum mismatch for $MODEL_URL" >&2; rm -f "$MODEL_WHL.part"; exit 1; }
  mv "$MODEL_WHL.part" "$MODEL_WHL"
fi

if [[ ! -x "$VENV/bin/python" ]]; then
  "$PY" -m venv "$VENV"
fi
# exact versions, no dependency resolution (scripts/spacy_export_requirements.txt)
"$VENV/bin/python" -m pip install --quiet --no-cache-dir --disable-pip-version-check --no-deps \
  -r "$HERE/spacy_export_requirements.txt" "$MODEL_WHL"
"$VENV/bin/python" "$HERE/export_spacy_assets.py" "$OUT"
(cd "$OUT" && echo "$EXPECTED" | sha256sum --check --quiet) || { echo "$0: exported files differ from the pinned hashes" >&2; exit 1; }
echo "ok (verified) $OUT"
