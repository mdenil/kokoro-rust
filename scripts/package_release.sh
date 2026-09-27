#!/usr/bin/env bash
# Build the release packages that install.sh downloads. Used by .github/workflows/release.yml and
# runnable locally with the same steps.
#
#   scripts/package_release.sh binary   VERSION OUT_DIR           # cargo build + binary package
#   scripts/package_release.sh frontend VERSION OUT_DIR DATA_DIR  # frontend data package
#   scripts/package_release.sh sums     OUT_DIR                   # SHA256SUMS of the packages
#
# binary:   OUT_DIR/kokoro-x86_64-linux.tar.gz, containing kokoro-x86_64-linux/
#             bin/kokoro, VERSION, LICENSE, README.md, THIRD_PARTY.md,
#             licenses/crates/<crate>-<version>/ (license files of the Rust crates used to build it),
#             licenses/CRATES.tsv (name, version, license expression, source)
# frontend: OUT_DIR/kokoro-frontend.tar.gz, containing frontend/
#             misaki-0.9.4/, spacy-en_core_web_sm-3.8.0/, VERSION, LICENSE, THIRD_PARTY.md,
#             licenses/ (misaki, spaCy, en_core_web_sm license files)
#           DATA_DIR must hold the output of scripts/fetch_assets.sh --frontend-only DATA_DIR and
#           scripts/prepare_spacy_assets.sh DATA_DIR (frontend/, downloads/, tools/spacy-export/).
# Needs bash, cargo (binary), python3 (binary: reads `cargo metadata`), tar, gzip, unzip, sha256sum.
# The archives are reproducible for the same inputs (sorted names, fixed owner and timestamps).
set -euo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
MTIME=@1700000000

die() { echo "$0: $*" >&2; exit 1; }

check_version() {
  [[ $1 =~ ^v[0-9A-Za-z._-]+$ ]] || die "version '$1' must look like v1.2.3"
}

archive() {  # archive SRC_PARENT NAME OUT_FILE
  tar -C "$1" --sort=name --owner=0 --group=0 --numeric-owner --mtime="$MTIME" --mode=u+rwX,go+rX,go-w \
    --pax-option=exthdr.name=%d/PaxHeaders/%f,delete=atime,delete=ctime -cf - "$2" | gzip -n -9 > "$3.part"
  mv "$3.part" "$3"
}

package_binary() {
  local version=$1 out=$2
  check_version "$version"
  mkdir -p "$out"
  (cd "$ROOT" && cargo build --release --locked)
  local stage
  stage=$(mktemp -d)
  trap 'rm -rf "$stage"' RETURN
  local pkg=$stage/kokoro-x86_64-linux
  mkdir -p "$pkg/bin" "$pkg/licenses/crates"
  install -m 755 "$ROOT/target/release/kokoro" "$pkg/bin/kokoro"
  echo "$version" > "$pkg/VERSION"
  cp "$ROOT/LICENSE" "$ROOT/README.md" "$ROOT/docs/THIRD_PARTY.md" "$pkg/"
  # license files of every crate used to build the Linux x86_64 binary (normal and build deps)
  (cd "$ROOT" && cargo metadata --format-version 1 --locked --filter-platform x86_64-unknown-linux-gnu) \
    | python3 "$ROOT/scripts/collect_crate_licenses.py" "$pkg/licenses"
  archive "$stage" kokoro-x86_64-linux "$out/kokoro-x86_64-linux.tar.gz"
  echo "wrote $out/kokoro-x86_64-linux.tar.gz"
}

package_frontend() {
  local version=$1 out=$2 data=$3
  check_version "$version"
  mkdir -p "$out"
  local misaki=$data/frontend/misaki-0.9.4 spacy=$data/frontend/spacy-en_core_web_sm-3.8.0
  local f
  for f in "$misaki/us_gold.json" "$misaki/us_silver.json" "$spacy/tokenizer.json" "$spacy/lookups.json" \
    "$spacy/symbols.json" "$spacy/base_norms.json" "$spacy/model_structure.json" "$spacy/tagger_weights.safetensors"; do
    [[ -f $f ]] || die "missing $f (run scripts/fetch_assets.sh --frontend-only and scripts/prepare_spacy_assets.sh on $data)"
  done
  local spacy_license
  spacy_license=$(ls "$data"/tools/spacy-export/lib/python3.12/site-packages/spacy-3.8.14.dist-info/licenses/LICENSE 2>/dev/null) \
    || die "spaCy 3.8.14 license file not found under $data/tools/spacy-export"
  local stage
  stage=$(mktemp -d)
  trap 'rm -rf "$stage"' RETURN
  local pkg=$stage/frontend
  mkdir -p "$pkg/licenses"
  cp -r "$misaki" "$spacy" "$pkg/"
  cp "$ROOT/LICENSE" "$ROOT/docs/THIRD_PARTY.md" "$pkg/"
  echo "$version" > "$pkg/VERSION"
  unzip -p "$data/downloads/misaki-0.9.4-py3-none-any.whl" misaki-0.9.4.dist-info/licenses/LICENSE > "$pkg/licenses/misaki-0.9.4-LICENSE"
  unzip -p "$data/downloads/en_core_web_sm-3.8.0-py3-none-any.whl" en_core_web_sm-3.8.0.dist-info/LICENSE > "$pkg/licenses/en_core_web_sm-3.8.0-LICENSE"
  unzip -p "$data/downloads/en_core_web_sm-3.8.0-py3-none-any.whl" en_core_web_sm-3.8.0.dist-info/LICENSES_SOURCES > "$pkg/licenses/en_core_web_sm-3.8.0-LICENSES_SOURCES"
  cp "$spacy_license" "$pkg/licenses/spacy-3.8.14-LICENSE"
  for f in "$pkg"/licenses/*; do
    [[ -s $f ]] || die "empty license file $f"
  done
  archive "$stage" frontend "$out/kokoro-frontend.tar.gz"
  echo "wrote $out/kokoro-frontend.tar.gz"
}

package_sums() {
  local out=$1
  (cd "$out" && sha256sum kokoro-x86_64-linux.tar.gz kokoro-frontend.tar.gz > SHA256SUMS)
  echo "wrote $out/SHA256SUMS"
  cat "$out/SHA256SUMS"
}

case ${1:-} in
  binary) [[ $# -eq 3 ]] || die "usage: $0 binary VERSION OUT_DIR"; package_binary "$2" "$3" ;;
  frontend) [[ $# -eq 4 ]] || die "usage: $0 frontend VERSION OUT_DIR DATA_DIR"; package_frontend "$2" "$3" "$4" ;;
  sums) [[ $# -eq 2 ]] || die "usage: $0 sums OUT_DIR"; package_sums "$2" ;;
  *) die "usage: $0 binary|frontend|sums ..." ;;
esac
