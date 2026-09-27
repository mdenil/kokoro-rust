#!/usr/bin/env bash
# Developer/test only: copy the eSpeak NG 1.52.0 library + data that the Python reference uses
# (bundled by espeakng_loader 0.2.4 in the reference venv) under the data root, with a sha256
# manifest. The tests that compare with the reference fixtures load this copy explicitly
# (tests/support/paths.rs); the product itself uses the system eSpeak NG. eSpeak NG is
# GPL-3.0-or-later and is NOT built into or shipped with this repo.
# usage: scripts/stage_espeak.sh [source espeakng_loader dir]
set -euo pipefail
source "$(dirname "$0")/env.sh"
src=${1:-$KOKORO_VENV/lib/python3.12/site-packages/espeakng_loader}
dst=$KOKORO_DATA/frontend/espeak-ng-1.52.0
mkdir -p "$dst"
cp -a "$src/libespeak-ng.so.1.52.0" "$dst/"
rm -rf "$dst/espeak-ng-data"
cp -a "$src/espeak-ng-data" "$dst/"
(cd "$dst" && find . -type f ! -name MANIFEST.sha256 -print0 | sort -z | xargs -0 sha256sum > MANIFEST.sha256)
echo "staged $(wc -l < "$dst/MANIFEST.sha256") files; manifest sha256 $(sha256sum "$dst/MANIFEST.sha256" | cut -c1-16)"
