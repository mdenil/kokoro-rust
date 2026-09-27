#!/usr/bin/env bash
# Wrapper around the skill's interleaved ABBA script that ALWAYS keeps the raw output as a receipt.
# usage: AB_PAIR_SCRIPT=/path/to/ab-pair.sh scripts/ab.sh <tag> -n N -g CV -- cmdA ::: cmdB
# (ab-pair.sh is the interleaved ABBA runner from the porting skill; it is not part of this repo)
set -euo pipefail
: "${AB_PAIR_SCRIPT:?set AB_PAIR_SCRIPT to the ab-pair.sh interleaved A/B runner}"
[[ -f "$AB_PAIR_SCRIPT" ]] || { echo "AB_PAIR_SCRIPT=$AB_PAIR_SCRIPT does not exist" >&2; exit 1; }
source "$(dirname "$0")/env.sh"
tag=$1; shift
dir="$KOKORO_DATA/evidence/ab"; mkdir -p "$dir"
out="$dir/$(date +%Y%m%d-%H%M%S)-$tag.txt"
{ echo "# tag=$tag git=$(git -C "$(dirname "$0")/.." rev-parse --short HEAD) host=$(uptime)"; echo "# cmd: $*"; } > "$out"
bash "$AB_PAIR_SCRIPT" "$@" 2>&1 | tee -a "$out"
sha256sum "$out" >> "$dir/SHA256SUMS"
echo "receipt: $out"
