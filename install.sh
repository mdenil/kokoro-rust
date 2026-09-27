#!/usr/bin/env bash
# kokoro installer (Linux x86_64, per user, no root needed):
#
#   curl -fsSL https://raw.githubusercontent.com/mdenil/kokoro-rust/main/install.sh | bash
#
# Installs the latest GitHub release of kokoro and the Kokoro-82M model:
#   ~/.local/bin/kokoro                              the command (a small launcher script)
#   ~/.local/share/kokoro/releases/<version>/        program and frontend data
#   ~/.local/share/kokoro/models/<model revision>/   model and voices, shared between versions
# ($XDG_DATA_HOME is used instead of ~/.local/share when set.)
#
# Environment variables (all optional):
#   KOKORO_VERSION       release tag to install (default: the latest release)
#   KOKORO_INSTALL_DIR   directory for the kokoro command (default: ~/.local/bin)
#   KOKORO_RELEASE_URL   base URL of the release files (a mirror; needs KOKORO_VERSION)
#   KOKORO_MODEL_URL     base URL of the model files (a mirror of the pinned Hugging Face revision)
#   KOKORO_TTY           terminal used to ask before installing eSpeak NG (default: /dev/tty)
#
# The NVIDIA driver and cuBLAS (CUDA 12) must already be installed; this script checks for them and
# never installs them. eSpeak NG is installed with apt-get only if you agree when asked.
# Every download is checked against a SHA-256 before it is used. A failed or interrupted run leaves
# an existing installation unchanged.

set -euo pipefail

REPO=mdenil/kokoro-rust
MODEL_REV=f3ff3571791e39611d31c381e3a41a3af07b4987
# model files: path, sha256 (hexgrad/Kokoro-82M at MODEL_REV)
MODEL_FILES="config.json 5abb01e2403b072bf03d04fde160443e209d7a0dad49a423be15196b9b43c17f
kokoro-v1_0.pth 496dba118d1a58f5f3db2efc88dbdc216e0483fc89fe6e47ee1f2c53f18ad1e4
voices/af_heart.pt 0ab5709b8ffab19bfd849cd11d98f75b60af7733253ad0d67b12382a102cb4ff
voices/am_adam.pt ced7e284aba12472891be1da3ab34db84cc05cc02b5889535796dbf2d8b0cb34"
BIN_ASSET=kokoro-x86_64-linux.tar.gz
FRONTEND_ASSET=kokoro-frontend.tar.gz
LAUNCHER_MARK="# kokoro launcher (written by install.sh)"

say() { printf '%s\n' "$*" >&2; }
die() { printf 'kokoro install: error: %s\n' "$*" >&2; exit 1; }

sha256_of() { sha256sum "$1" | cut -d' ' -f1; }

download() {  # download URL DEST
  curl -fL --retry 3 --silent --show-error -o "$2" "$1" || die "download failed: $1"
}

ldconfig_cmd() { PATH="$PATH:/sbin:/usr/sbin" command -v ldconfig || true; }

have_lib() {  # have_lib SONAME: can the dynamic loader find this x86_64 library?
  local d ldc
  if [[ -n ${LD_LIBRARY_PATH:-} ]]; then
    local IFS=:
    for d in $LD_LIBRARY_PATH; do
      [[ -n $d && -e $d/$1 ]] && return 0
    done
  fi
  ldc=$(ldconfig_cmd)
  if [[ -n $ldc ]]; then
    "$ldc" -p 2>/dev/null | awk -v n="$1" '$1 == n && /x86-64/ { found = 1 } END { exit !found }'
    return
  fi
  for d in /lib/x86_64-linux-gnu /usr/lib/x86_64-linux-gnu /lib64 /usr/lib64; do
    [[ -e $d/$1 ]] && return 0
  done
  return 1
}

check_platform() {
  local os arch
  os=$(uname -s)
  arch=$(uname -m)
  [[ $os == Linux && $arch == x86_64 ]] || die "kokoro runs on Linux x86_64 only (this system: $os $arch)"
  local tool
  for tool in curl tar gzip sha256sum mktemp awk; do
    command -v "$tool" > /dev/null || die "'$tool' is required but not installed"
  done
}

check_gpu_libraries() {
  local missing=()
  have_lib libcuda.so.1 || missing+=("libcuda.so.1 (NVIDIA driver)")
  have_lib libnvidia-ptxjitcompiler.so.1 || missing+=("libnvidia-ptxjitcompiler.so.1 (NVIDIA driver)")
  have_lib libcublas.so.12 || missing+=("libcublas.so.12 (cuBLAS, CUDA 12)")
  have_lib libcublasLt.so.12 || missing+=("libcublasLt.so.12 (cuBLAS, CUDA 12)")
  [[ ${#missing[@]} -eq 0 ]] && return
  say "These GPU libraries were not found by the dynamic loader:"
  local m
  for m in "${missing[@]}"; do say "  - $m"; done
  say "kokoro needs an NVIDIA GPU with compute capability 8.9 or newer, an NVIDIA driver that"
  say "supports CUDA 12.9, and cuBLAS from CUDA 12 (for example NVIDIA's libcublas-12-9 package or"
  say "the CUDA Toolkit; the rest of the toolkit is not needed). If they are installed in a"
  say "non-standard directory, add it to LD_LIBRARY_PATH. This installer does not install them."
  die "missing GPU libraries"
}

ensure_espeak() {
  have_lib libespeak-ng.so.1 && return
  say "eSpeak NG (libespeak-ng.so.1) is not installed. kokoro needs it to read text: it pronounces"
  say "words that are not in its dictionary."
  if ! command -v apt-get > /dev/null; then
    die "install eSpeak NG with your package manager (the package providing libespeak-ng.so.1), then run this installer again"
  fi
  local manual="sudo apt-get install libespeak-ng1"
  local tty=${KOKORO_TTY:-/dev/tty}
  if ! { exec 3< "$tty"; } 2> /dev/null; then
    die "no terminal to ask for permission; install it with '$manual', then run this installer again"
  fi
  printf "Install it now with '%s'? sudo may ask for your password. [y/N] " "$manual" >&2
  local answer=""
  read -r answer <&3 || true
  [[ -t 3 ]] || printf '\n' >&2
  exec 3>&-
  case $answer in
    y | Y | yes | Yes | YES) ;;
    *) die "eSpeak NG not installed; install it with '$manual', then run this installer again" ;;
  esac
  # apt-get must not read the rest of a piped installer from stdin (the redirect is intentionally
  # the user's own terminal, not a privileged file)
  # shellcheck disable=SC2024
  sudo apt-get install -y libespeak-ng1 < "$tty" || die "'sudo apt-get install -y libespeak-ng1' failed"
  have_lib libespeak-ng.so.1 || die "libespeak-ng.so.1 is still not found after installing libespeak-ng1"
  say "eSpeak NG installed."
}

resolve_version() {
  if [[ -n ${KOKORO_VERSION:-} ]]; then
    printf '%s\n' "$KOKORO_VERSION"
    return
  fi
  [[ -z ${KOKORO_RELEASE_URL:-} ]] || die "KOKORO_RELEASE_URL needs KOKORO_VERSION"
  local url
  url=$(curl -fsSLI -o /dev/null -w '%{url_effective}' "https://github.com/$REPO/releases/latest") \
    || die "could not look up the latest release at https://github.com/$REPO/releases (no published release, or no network access); KOKORO_VERSION selects a version"
  case $url in
    */releases/tag/*) printf '%s\n' "${url##*/releases/tag/}" ;;
    *) die "no published release found at https://github.com/$REPO/releases" ;;
  esac
}

# SHA256SUMS is data: only well-formed lines for the two expected files are used
expected_sha() {  # expected_sha SUMS_FILE NAME
  awk -v n="$2" 'length($1) == 64 && $1 ~ /^[0-9a-f]+$/ && NF == 2 && $2 == n { print $1; c++ } END { exit c != 1 }' "$1" \
    || die "SHA256SUMS of the release has no valid entry for $2"
}

safe_archive() {  # safe_archive TARBALL PREFIX: every member is a relative path under PREFIX/
  tar -tzf "$1" | awk -v p="$2/" '
    { if (substr($0, 1, length(p)) != p && $0 != p) bad = 1; if ($0 ~ /(^|\/)\.\.(\/|$)/) bad = 1 }
    END { exit bad }' || die "unexpected paths in $(basename "$1")"
}

# Globals of one run: STAGING (temporary directory), RELEASE_DIR (target of this version),
# NEW_RELEASE (verified new copy waiting in STAGING, empty when the installed copy is reused) and
# PREVIOUS (the replaced copy of the same version, kept until the whole install has succeeded).
STAGING=""
RELEASE_DIR=""
NEW_RELEASE=""
PREVIOUS=""

cleanup() {  # EXIT trap: an unfinished install puts the previous copy of this version back
  local rc=$?
  if [[ -n $PREVIOUS && -d $PREVIOUS ]]; then
    if [[ -e $RELEASE_DIR ]]; then
      mv "$RELEASE_DIR" "$STAGING/discarded-release" 2> /dev/null || rm -rf "$RELEASE_DIR"
    fi
    mv "$PREVIOUS" "$RELEASE_DIR" && say "The previous installation of this version was restored."
  fi
  [[ -z $STAGING ]] || rm -rf "$STAGING"
  exit "$rc"
}

recover_previous() {  # a run killed outright (SIGKILL, power loss) may leave releases/.previous-<version>
  local prev=$1/releases/.previous-$2 target=$1/releases/$2
  [[ -d $prev ]] || return 0
  if [[ -e $target ]]; then
    rm -rf "$prev"  # the new copy had been moved in completely
  else
    mv "$prev" "$target"
    say "Restored the previous installation of $2 left by an interrupted run."
  fi
}

stage_release() {  # stage_release VERSION BASE_URL DATA: download, verify and unpack into STAGING
  local version=$1 base=$2 data=$3 staging=$STAGING
  RELEASE_DIR=$data/releases/$version
  local dl=$staging/download
  mkdir -p "$dl"
  download "$base/SHA256SUMS" "$dl/SHA256SUMS"
  local bin_sha front_sha
  bin_sha=$(expected_sha "$dl/SHA256SUMS" "$BIN_ASSET")
  front_sha=$(expected_sha "$dl/SHA256SUMS" "$FRONTEND_ASSET")
  if [[ -f $RELEASE_DIR/.installed && -f $RELEASE_DIR/.SHA256SUMS ]] \
    && cmp -s "$dl/SHA256SUMS" "$RELEASE_DIR/.SHA256SUMS" \
    && (cd "$RELEASE_DIR" && sha256sum --quiet --strict -c .files.sha256 > /dev/null 2>&1); then
    say "Release $version is already installed and intact."
    return
  fi
  say "Downloading kokoro $version ..."
  download "$base/$BIN_ASSET" "$dl/$BIN_ASSET"
  download "$base/$FRONTEND_ASSET" "$dl/$FRONTEND_ASSET"
  [[ $(sha256_of "$dl/$BIN_ASSET") == "$bin_sha" ]] || die "checksum mismatch for $BIN_ASSET"
  [[ $(sha256_of "$dl/$FRONTEND_ASSET") == "$front_sha" ]] || die "checksum mismatch for $FRONTEND_ASSET"
  safe_archive "$dl/$BIN_ASSET" kokoro-x86_64-linux
  safe_archive "$dl/$FRONTEND_ASSET" frontend
  local new=$staging/release
  mkdir -p "$new"
  tar -xzf "$dl/$BIN_ASSET" -C "$new" --strip-components=1 --no-same-owner
  tar -xzf "$dl/$FRONTEND_ASSET" -C "$new" --no-same-owner
  local f
  for f in bin/kokoro frontend/misaki-0.9.4/us_gold.json frontend/spacy-en_core_web_sm-3.8.0/tagger_weights.safetensors; do
    [[ -f $new/$f ]] || die "release package is missing $f"
  done
  "$new/bin/kokoro" --version > /dev/null 2>&1 \
    || die "the downloaded kokoro binary does not run on this system (it needs glibc 2.35 or newer)"
  (cd "$new" && find . -type f ! -name '.files.sha256' -print0 | LC_ALL=C sort -z | xargs -0 sha256sum > .files.sha256)
  cp "$dl/SHA256SUMS" "$new/.SHA256SUMS"
  touch "$new/.installed"
  NEW_RELEASE=$new
}

commit_release() {  # move the staged release into place; a replaced copy is kept as PREVIOUS
  [[ -n $NEW_RELEASE ]] || return 0
  mkdir -p "$(dirname "$RELEASE_DIR")"
  if [[ -e $RELEASE_DIR ]]; then
    PREVIOUS=$(dirname "$RELEASE_DIR")/.previous-$(basename "$RELEASE_DIR")
    rm -rf "$PREVIOUS"
    mv "$RELEASE_DIR" "$PREVIOUS"
  fi
  mv "$NEW_RELEASE" "$RELEASE_DIR"
}

finish_release() {  # the launcher is written: the replaced copy is no longer needed
  if [[ -n $PREVIOUS ]]; then
    local prev=$PREVIOUS
    PREVIOUS=""
    rm -rf "$prev"
  fi
}

install_model() {  # install_model BASE_URL DATA -> sets MODEL_DIR
  local base=$1 data=$2 staging=$STAGING
  MODEL_DIR=$data/models/$MODEL_REV
  local path sha dest tmp
  while read -r path sha; do
    dest=$MODEL_DIR/$path
    if [[ -f $dest && $(sha256_of "$dest") == "$sha" ]]; then
      continue
    fi
    say "Downloading model file $path ..."
    tmp=$staging/model/$path
    mkdir -p "$(dirname "$tmp")" "$(dirname "$dest")"
    download "$base/$path" "$tmp"
    [[ $(sha256_of "$tmp") == "$sha" ]] || die "checksum mismatch for model file $path"
    mv -f "$tmp" "$dest"
  done <<< "$MODEL_FILES"
  say "Model $MODEL_REV is in place."
}

check_launcher_target() {  # never replace a kokoro command this installer did not write
  local target=$1/kokoro
  if [[ -e $target || -L $target ]] && ! grep -qxF "$LAUNCHER_MARK" "$target" 2> /dev/null; then
    die "$target exists and was not written by this installer; move it away or set KOKORO_INSTALL_DIR"
  fi
}

write_launcher() {  # write_launcher BIN_DIR RELEASE_DIR MODEL_DIR
  local bin_dir=$1 release=$2 model=$3 target=$1/kokoro
  mkdir -p "$bin_dir"
  check_launcher_target "$bin_dir"
  local tmp
  tmp=$(mktemp "$bin_dir/.kokoro.XXXXXX")
  cat > "$tmp" << EOF
#!/bin/sh
$LAUNCHER_MARK
# Defaults for the installed data; --model-dir / --frontend-dir or these variables override them.
if [ -z "\${KOKORO_MODEL_DIR:-}" ]; then KOKORO_MODEL_DIR='$model'; fi
if [ -z "\${KOKORO_FRONTEND_DIR:-}" ]; then KOKORO_FRONTEND_DIR='$release/frontend'; fi
export KOKORO_MODEL_DIR KOKORO_FRONTEND_DIR
exec '$release/bin/kokoro' "\$@"
EOF
  chmod 755 "$tmp"
  mv -f "$tmp" "$target" || { rm -f "$tmp"; die "could not write $target"; }
}

main() {
  check_platform
  check_gpu_libraries
  ensure_espeak

  local data=${XDG_DATA_HOME:-$HOME/.local/share}/kokoro
  local bin_dir=${KOKORO_INSTALL_DIR:-$HOME/.local/bin}
  local version
  version=$(resolve_version)
  [[ $version =~ ^v[0-9A-Za-z._-]+$ ]] || die "unexpected release version '$version'"
  local p
  for p in "$data" "$bin_dir"; do
    [[ $p != *"'"* ]] || die "installation path contains a single quote: $p"
  done
  check_launcher_target "$bin_dir"
  local release_base=${KOKORO_RELEASE_URL:-https://github.com/$REPO/releases/download/$version}
  local model_base=${KOKORO_MODEL_URL:-https://huggingface.co/hexgrad/Kokoro-82M/resolve/$MODEL_REV}

  mkdir -p "$data"
  recover_previous "$data" "$version"
  trap cleanup EXIT
  trap 'exit 130' INT TERM HUP
  STAGING=$(mktemp -d "$data/.install.XXXXXX")

  # nothing installed changes until the release and the model are verified; a replaced copy of the
  # same version is restored if the remaining steps fail
  stage_release "$version" "$release_base" "$data"
  install_model "$model_base" "$data"
  commit_release
  write_launcher "$bin_dir" "$RELEASE_DIR" "$MODEL_DIR"
  finish_release

  say ""
  say "kokoro $version installed: $bin_dir/kokoro"
  case ":$PATH:" in
    *":$bin_dir:"*) ;;
    *) say "Note: $bin_dir is not on your PATH; add it (for example in ~/.profile) or run $bin_dir/kokoro." ;;
  esac
  local other
  other=$(find "$data/releases" -mindepth 1 -maxdepth 1 -type d ! -name "$version" ! -name '.*' -printf '%f ' 2> /dev/null || true)
  [[ -z $other ]] || say "Other installed versions (can be deleted from $data/releases): $other"
  say "Try: kokoro synth book.txt   (writes book.wav in the current directory)"
  if ! command -v ffmpeg > /dev/null; then
    say "(Optional: install ffmpeg to use --encode for FLAC/MP3/Opus output.)"
  fi
}

main "$@"
