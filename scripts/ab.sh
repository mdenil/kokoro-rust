#!/usr/bin/env bash
# Wrapper around the skill's interleaved ABBA script that ALWAYS keeps the raw output as a receipt.
# usage: scripts/ab.sh <tag> -n N -g CV -- cmdA ::: cmdB
set -euo pipefail
source "$(dirname "$0")/env.sh"
tag=$1; shift
dir="$KOKORO_DATA/evidence/ab"; mkdir -p "$dir"
out="$dir/$(date +%Y%m%d-%H%M%S)-$tag.txt"
{ echo "# tag=$tag git=$(git -C "$(dirname "$0")/.." rev-parse --short HEAD) host=$(uptime)"; echo "# cmd: $*"; } > "$out"
bash /home/mdenil/.claude/skills/ai-model-into-rust-mega-fused-hyper-kernel/scripts/ab-pair.sh "$@" 2>&1 | tee -a "$out"
sha256sum "$out" >> "$dir/SHA256SUMS"
echo "receipt: $out"
