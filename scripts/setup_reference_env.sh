#!/usr/bin/env bash
# Recreate the pinned production-reference Python environment (oracle side).
# Pins are the versions observed in production use (docs/truth-pack/PINNED_SOURCES.md); the oracle asserts
# them at runtime. Everything large lives under the data root $KOKORO_DATA.
# The interpreter must be CPython 3.12.3 (asserted below): $REFERENCE_PYTHON, default python3.12.
set -euo pipefail

source "$(dirname "$0")/env.sh"
DATA="$KOKORO_DATA"
VENV="$DATA/reference/venv-prod"

uv venv --python "${REFERENCE_PYTHON:-python3.12}" "$VENV"
uv pip install -p "$VENV/bin/python" \
  torch==2.12.1 numpy==2.4.6 soundfile==0.14.0 transformers==5.12.1 \
  espeakng-loader==0.2.4 spacy==3.8.14 huggingface-hub==1.20.1 \
  kokoro==0.9.4 'misaki[en]==0.9.4' \
  'en-core-web-sm @ https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-3.8.0/en_core_web_sm-3.8.0-py3-none-any.whl'

"$VENV/bin/python" - <<'EOF'
import sys, importlib.metadata as md
assert sys.version_info[:3] == (3, 12, 3), sys.version
for pkg, want in [("torch","2.12.1"),("numpy","2.4.6"),("soundfile","0.14.0"),
                  ("transformers","5.12.1"),("espeakng-loader","0.2.4"),("spacy","3.8.14"),
                  ("huggingface-hub","1.20.1"),("kokoro","0.9.4"),("misaki","0.9.4"),
                  ("en_core_web_sm","3.8.0")]:
    got = md.version(pkg)
    assert got == want, f"{pkg}: got {got}, want {want}"
print("reference env pins OK")
EOF
