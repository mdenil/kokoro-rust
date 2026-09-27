#!/usr/bin/env bash
# Download the public runtime assets from their pinned origins into DATA_DIR and verify every file
# against a pinned SHA-256. Needs only bash, curl, unzip and sha256sum (no Python).
#
#   scripts/fetch_assets.sh DATA_DIR
#
# Result:
#   DATA_DIR/model/                             Kokoro-82M config, weights, voices af_heart + am_adam
#   DATA_DIR/frontend/misaki-0.9.4/             misaki 0.9.4 US lexicons (from the misaki wheel)
#   DATA_DIR/frontend/espeak-ng-1.52.0/         libespeak-ng 1.52.0 + espeak-ng-data (from the
#                                               espeakng-loader 0.2.4 manylinux x86_64 wheel)
#   DATA_DIR/downloads/                         the downloaded wheels
# The spaCy part of the frontend data needs a Python step: scripts/prepare_spacy_assets.sh.
# Re-running is safe: files that already verify are not downloaded again.
set -euo pipefail

if [[ $# -ne 1 ]]; then
  echo "usage: $0 DATA_DIR" >&2
  exit 2
fi
DATA=$1
for tool in curl unzip sha256sum; do
  command -v "$tool" >/dev/null || { echo "$0: '$tool' is required but not on PATH" >&2; exit 2; }
done
mkdir -p "$DATA"/{model/voices,frontend,downloads}

REV=f3ff3571791e39611d31c381e3a41a3af07b4987
HF="https://huggingface.co/hexgrad/Kokoro-82M/resolve/$REV"
MISAKI_WHL=misaki-0.9.4-py3-none-any.whl
MISAKI_URL=https://files.pythonhosted.org/packages/82/ec/0ee4110ddb54278b8f21c40a140370ae8f687036c4edf578316602697c56/$MISAKI_WHL
ESPEAK_WHL=espeakng_loader-0.2.4-py3-none-manylinux_2_17_x86_64.manylinux2014_x86_64.whl
ESPEAK_URL=https://files.pythonhosted.org/packages/de/1e/25ec5ab07528c0fbb215a61800a38eca05c8a99445515a02d7fa5debcb32/$ESPEAK_WHL

verify() {  # verify FILE SHA256 -> 0 if the file exists with that hash
  [[ -f "$1" ]] && [[ "$(sha256sum "$1" | cut -d' ' -f1)" == "$2" ]]
}

fetch() {  # fetch URL DEST SHA256
  local url=$1 dest=$2 sha=$3
  if verify "$dest" "$sha"; then
    echo "ok (present)  $dest"
    return
  fi
  curl -fL --retry 3 --silent --show-error -o "$dest.part" "$url"
  if ! verify "$dest.part" "$sha"; then
    echo "$0: checksum mismatch for $url (expected $sha); not installed" >&2
    rm -f "$dest.part"
    exit 1
  fi
  mv "$dest.part" "$dest"
  echo "ok (fetched)  $dest"
}

extract() {  # extract WHEEL MEMBER DEST SHA256 (single file)
  local whl=$1 member=$2 dest=$3 sha=$4
  verify "$dest" "$sha" && { echo "ok (present)  $dest"; return; }
  mkdir -p "$(dirname "$dest")"
  unzip -p "$whl" "$member" > "$dest.part"
  verify "$dest.part" "$sha" || { echo "$0: unexpected content for $member in $whl" >&2; rm -f "$dest.part"; exit 1; }
  mv "$dest.part" "$dest"
  echo "ok (extracted) $dest"
}

# --- model: hexgrad/Kokoro-82M at revision $REV (Apache-2.0)
fetch "$HF/config.json"          "$DATA/model/config.json"          5abb01e2403b072bf03d04fde160443e209d7a0dad49a423be15196b9b43c17f
fetch "$HF/kokoro-v1_0.pth"      "$DATA/model/kokoro-v1_0.pth"      496dba118d1a58f5f3db2efc88dbdc216e0483fc89fe6e47ee1f2c53f18ad1e4
fetch "$HF/voices/af_heart.pt"   "$DATA/model/voices/af_heart.pt"   0ab5709b8ffab19bfd849cd11d98f75b60af7733253ad0d67b12382a102cb4ff
fetch "$HF/voices/am_adam.pt"    "$DATA/model/voices/am_adam.pt"    ced7e284aba12472891be1da3ab34db84cc05cc02b5889535796dbf2d8b0cb34

# --- misaki 0.9.4 lexicons (Apache-2.0), from the PyPI wheel
fetch "$MISAKI_URL" "$DATA/downloads/$MISAKI_WHL" 90e2eeb169786c014c429e5058d2ea6bcd02d651f2a24450ba6c9ffc0f8da15a
extract "$DATA/downloads/$MISAKI_WHL" misaki/data/us_gold.json   "$DATA/frontend/misaki-0.9.4/us_gold.json"   dc414872a49a28ae6c141463d502fd945f3b2fde040484fdc47d00cc4612686f
extract "$DATA/downloads/$MISAKI_WHL" misaki/data/us_silver.json "$DATA/frontend/misaki-0.9.4/us_silver.json" de8f67be911bb6c659187b4a65fd966b6a30e56350e0f790d763210b053ac475

# --- espeak-ng 1.52.0 library + data (GPL-3.0-or-later), from the espeakng-loader 0.2.4 wheel
fetch "$ESPEAK_URL" "$DATA/downloads/$ESPEAK_WHL" 08721baf27d13d461f6be6eed9a65277e70d68234ff484fd8b9897b222cdcb6d
ES="$DATA/frontend/espeak-ng-1.52.0"
extract "$DATA/downloads/$ESPEAK_WHL" espeakng_loader/libespeak-ng.so.1.52.0 "$ES/libespeak-ng.so.1.52.0" b15ce0803b26254e8a0d088c6b9281a2637b2b02613147cc320cfdefaf2c3ccf
ESPEAK_DATA_DIGEST=3c70a30029ed60daf705682bf2991611c0cca49b4158b75f2ec7abaa7e34ef29
tree_digest() { (cd "$1" && find espeak-ng-data -type f -print0 | LC_ALL=C sort -z | xargs -0 sha256sum | sha256sum | cut -d' ' -f1); }
if [[ -d "$ES/espeak-ng-data" && "$(tree_digest "$ES")" == "$ESPEAK_DATA_DIGEST" ]]; then
  echo "ok (present)  $ES/espeak-ng-data"
else
  tmp=$(mktemp -d "$DATA/downloads/unpack.XXXXXX")
  unzip -q "$DATA/downloads/$ESPEAK_WHL" 'espeakng_loader/espeak-ng-data/*' -d "$tmp"
  [[ "$(tree_digest "$tmp/espeakng_loader")" == "$ESPEAK_DATA_DIGEST" ]] || { echo "$0: unexpected espeak-ng-data content in $ESPEAK_WHL" >&2; rm -rf "$tmp"; exit 1; }
  rm -rf "$ES/espeak-ng-data"
  mv "$tmp/espeakng_loader/espeak-ng-data" "$ES/espeak-ng-data"
  rm -rf "$tmp"
  echo "ok (extracted) $ES/espeak-ng-data (364 files)"
fi

echo
echo "Model:    $DATA/model"
echo "Frontend: $DATA/frontend (run scripts/prepare_spacy_assets.sh $DATA to add the spaCy data)"
