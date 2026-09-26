#!/usr/bin/env bash
# Fetch (and with --verify, hash-check) the pinned Kokoro-82M snapshot.
set -euo pipefail

source "$(dirname "$0")/env.sh"
DATA="$KOKORO_DATA"
REV=f3ff3571791e39611d31c381e3a41a3af07b4987
SNAP="$DATA/hf/hub/models--hexgrad--Kokoro-82M/snapshots/$REV"
# HF_HOME / UV_CACHE_DIR come from env.sh

uvx --from 'huggingface-hub==1.20.1' hf download hexgrad/Kokoro-82M --revision "$REV" >/dev/null
echo "snapshot at: $SNAP"

if [[ "${1:-}" == "--verify" ]]; then
  cd "$SNAP"
  sha256sum -c - <<'EOF'
5abb01e2403b072bf03d04fde160443e209d7a0dad49a423be15196b9b43c17f  config.json
496dba118d1a58f5f3db2efc88dbdc216e0483fc89fe6e47ee1f2c53f18ad1e4  kokoro-v1_0.pth
0ab5709b8ffab19bfd849cd11d98f75b60af7733253ad0d67b12382a102cb4ff  voices/af_heart.pt
ced7e284aba12472891be1da3ab34db84cc05cc02b5889535796dbf2d8b0cb34  voices/am_adam.pt
EOF
  echo "pinned hashes verified"
fi
