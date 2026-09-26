#!/usr/bin/env bash
# Recreate the pinned production-reference Python environment (oracle side).
# Pins are the OBSERVED production versions from HERMES_BRIEF.md; the oracle asserts
# them at runtime. Everything large lives under /data/mdenil/kokoro-rust.
set -euo pipefail

DATA=${KOKORO_DATA:-/data/mdenil/kokoro-rust}
VENV="$DATA/reference/venv-prod"
export UV_CACHE_DIR="$DATA/uv-cache"

uv venv --python /usr/bin/python3 "$VENV"
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
