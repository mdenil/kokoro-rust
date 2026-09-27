#!/usr/bin/env bash
# Scenario tests for install.sh, run from a checkout (developer tool; not part of `cargo test`).
#
#   tests/installer/test_install.sh RELEASES_DIR WORK_DIR
#
# RELEASES_DIR holds locally built release packages (scripts/package_release.sh):
#   RELEASES_DIR/dist1/  version v0.1.0-test1: kokoro-x86_64-linux.tar.gz, kokoro-frontend.tar.gz, SHA256SUMS
#   RELEASES_DIR/dist2/  version v0.1.0-test2
# They are served to the installer through KOKORO_RELEASE_URL=file://... (a local stand-in for
# GitHub Releases). The first fresh install downloads the model from Hugging Face; later scenarios
# use a file:// copy of that verified download (KOKORO_MODEL_URL).
# sudo, apt-get, ldconfig and uname are replaced by MOCKS on PATH where a scenario says so: no real
# privileged command or package installation is ever run. The installer always runs piped into
# bash (curl file://.../install.sh | bash), in an empty environment with a fresh HOME.
# Needs an NVIDIA GPU for the synthesis checks (CUDA_VISIBLE_DEVICES is passed through).
set -uo pipefail

REPO=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
RELEASES=${1:?usage: $0 RELEASES_DIR WORK_DIR}
WORK=${2:?usage: $0 RELEASES_DIR WORK_DIR}
V1=v0.1.0-test1
V2=v0.1.0-test2
REV=f3ff3571791e39611d31c381e3a41a3af07b4987
REAL_LDCONFIG=$(PATH="$PATH:/sbin:/usr/sbin" command -v ldconfig)
FAILS=0
PASSES=0

rm -rf "$WORK"
mkdir -p "$WORK"
MOCKS=$WORK/mocks
mkdir -p "$MOCKS"

# --- mocks ------------------------------------------------------------------------------------
cat > "$MOCKS/ldconfig" << EOF
#!/bin/bash
# MOCK ldconfig: the real cache, minus libraries a scenario hides
"$REAL_LDCONFIG" "\$@" | {
  if [[ -n \${MOCK_HIDE_ESPEAK:-} && ! -e \$MOCK_STATE/espeak_installed ]]; then grep -v 'libespeak-ng\.so\.1 '; else cat; fi
} | {
  if [[ -n \${MOCK_HIDE_CUBLAS:-} ]]; then grep -v 'libcublas\.so\.12 '; else cat; fi
}
EOF
cat > "$MOCKS/sudo" << 'EOF'
#!/bin/bash
# MOCK sudo: records the command and runs it unprivileged (it only ever reaches the apt-get mock)
echo "sudo $*" >> "$MOCK_STATE/sudo.log"
exec "$@"
EOF
cat > "$MOCKS/apt-get" << 'EOF'
#!/bin/bash
# MOCK apt-get: records arguments and where stdin comes from; "installs" by flipping a state file
echo "apt-get $* ; stdin=$(readlink /proc/self/fd/0)" >> "$MOCK_STATE/apt.log"
[[ ${MOCK_APT_EXIT:-0} == 0 ]] || exit "$MOCK_APT_EXIT"
[[ $* == *"install -y libespeak-ng1"* ]] && touch "$MOCK_STATE/espeak_installed"
exit 0
EOF
cat > "$MOCKS/uname" << 'EOF'
#!/bin/bash
# MOCK uname: reports an aarch64 machine
case ${1:-} in -m) echo aarch64 ;; *) echo Linux ;; esac
EOF
chmod +x "$MOCKS"/*
mkdir -p "$MOCKS/no-uname" "$MOCKS/std"
for m in ldconfig sudo apt-get; do ln -s "$MOCKS/$m" "$MOCKS/std/$m"; done

# --- helpers ----------------------------------------------------------------------------------
check() {  # check DESCRIPTION COMMAND...
  if "${@:2}"; then
    PASSES=$((PASSES + 1)); echo "  ok   $1"
  else
    FAILS=$((FAILS + 1)); echo "  FAIL $1"
  fi
}

# run_install NAME PREFIX [VAR=value ...]: run `curl file://install.sh | bash` in an empty environment
# (HOME=PREFIX, PATH = MOCK dir (if MOCKPATH is given) + /usr/bin:/bin); sets RC, OUT (stderr+stdout)
run_install() {
  local name=$1 prefix=$2
  shift 2
  mkdir -p "$prefix" "$WORK/state/$name"
  local log=$WORK/$name.log
  curl -fsSL "file://$REPO/install.sh" | env -i HOME="$prefix" PATH="${MOCKPATH:+$MOCKPATH:}/usr/bin:/bin" \
    MOCK_STATE="$WORK/state/$name" CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}" "$@" bash > "$log" 2>&1
  RC=${PIPESTATUS[1]}
  OUT=$(cat "$log")
  echo "[$name] exit=$RC ($log)"
}

has() { grep -qF -- "$1" <<< "$OUT"; }
no_file() { [[ ! -e $1 ]]; }
launcher_points_to() { grep -qF "$2/bin/kokoro" "$1/.local/bin/kokoro"; }
no_staging() { ! compgen -G "$1/.install.*" > /dev/null; }

synth() {  # synth PREFIX NAME [args...]: the installed command, no paths given
  local prefix=$1 name=$2
  shift 2
  printf 'Hello from the installed kokoro command.\n' > "$WORK/$name.txt"
  env -i HOME="$prefix" PATH="$prefix/.local/bin:/usr/bin:/bin" CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}" \
    kokoro synth --input "$WORK/$name.txt" --out-dir "$WORK/$name-out" "$@" > "$WORK/$name.synth.log" 2>&1
}

URL1="file://$RELEASES/dist1"
URL2="file://$RELEASES/dist2"
MIRROR=$WORK/model-mirror

# ================================================================================================
echo "S1 fresh install, piped, real Hugging Face model download"
P=$WORK/home-main
MOCKPATH='' run_install s1 "$P" KOKORO_VERSION=$V1 KOKORO_RELEASE_URL="$URL1"
D=$P/.local/share/kokoro
check "exit 0" test "$RC" = 0
check "launcher in ~/.local/bin" test -x "$P/.local/bin/kokoro"
check "launcher points to release $V1" launcher_points_to "$P" "$D/releases/$V1"
check "model from Hugging Face in models/<rev>" test -f "$D/models/$REV/kokoro-v1_0.pth"
check "model file downloads logged" has "Downloading model file kokoro-v1_0.pth"
check "no staging left" no_staging "$D"
check "PATH hint printed (bin dir not on PATH)" has "is not on your PATH"
synth "$P" s1
check "installed command synthesizes with no path options" test -s "$WORK/s1-out/s1_00001.wav"
mkdir -p "$MIRROR"
cp -r "$D/models/$REV/." "$MIRROR/"
MIRROR_URL="file://$MIRROR"

echo "S2 rerun, same version: everything reused"
ino_rel=$(stat -c %i "$D/releases/$V1")
ino_model=$(stat -c %i "$D/models/$REV/kokoro-v1_0.pth")
MOCKPATH='' run_install s2 "$P" KOKORO_VERSION=$V1 KOKORO_RELEASE_URL="$URL1"
check "exit 0" test "$RC" = 0
check "release reused" has "already installed and intact"
check "release directory untouched" test "$(stat -c %i "$D/releases/$V1")" = "$ino_rel"
check "no model download" bash -c "! grep -q 'Downloading model' '$WORK/s2.log'"
check "model file untouched" test "$(stat -c %i "$D/models/$REV/kokoro-v1_0.pth")" = "$ino_model"

echo "S3 upgrade to $V2"
MOCKPATH='' run_install s3 "$P" KOKORO_VERSION=$V2 KOKORO_RELEASE_URL="$URL2"
check "exit 0" test "$RC" = 0
check "launcher points to $V2" launcher_points_to "$P" "$D/releases/$V2"
check "previous version kept" test -d "$D/releases/$V1"
check "previous version listed" has "Other installed versions"
check "model reused" bash -c "! grep -q 'Downloading model' '$WORK/s3.log'"
synth "$P" s3
check "upgraded command synthesizes" test -s "$WORK/s3-out/s3_00001.wav"

echo "S4 damaged installed release is repaired on rerun"
rm "$D/releases/$V2/frontend/misaki-0.9.4/us_gold.json"
MOCKPATH='' run_install s4 "$P" KOKORO_VERSION=$V2 KOKORO_RELEASE_URL="$URL2"
check "exit 0" test "$RC" = 0
check "release downloaded again" has "Downloading kokoro $V2"
check "file restored" test -f "$D/releases/$V2/frontend/misaki-0.9.4/us_gold.json"

echo "S5 corrupt package: failure leaves the working install unchanged"
cp "$P/.local/bin/kokoro" "$WORK/launcher.before"
BAD=$RELEASES/dist-corrupt
rm -rf "$BAD"; cp -r "$RELEASES/dist2" "$BAD"
printf 'X' | dd of="$BAD/kokoro-frontend.tar.gz" bs=1 seek=1000 conv=notrunc status=none
MOCKPATH='' run_install s5 "$P" KOKORO_VERSION=v0.1.0-test3 KOKORO_RELEASE_URL="file://$BAD"
check "exit non-zero" test "$RC" != 0
check "checksum mismatch reported" has "checksum mismatch for kokoro-frontend.tar.gz"
check "launcher unchanged" cmp -s "$WORK/launcher.before" "$P/.local/bin/kokoro"
check "no partial release" no_file "$D/releases/v0.1.0-test3"
check "no staging left" no_staging "$D"

echo "S6 missing package file (interrupted/incomplete release): install unchanged"
rm -f "$BAD/kokoro-frontend.tar.gz"
cp "$RELEASES/dist2/SHA256SUMS" "$BAD/SHA256SUMS"
MOCKPATH='' run_install s6 "$P" KOKORO_VERSION=v0.1.0-test3 KOKORO_RELEASE_URL="file://$BAD"
check "exit non-zero" test "$RC" != 0
check "download failure reported" has "download failed"
check "launcher unchanged" cmp -s "$WORK/launcher.before" "$P/.local/bin/kokoro"
check "no staging left" no_staging "$D"

echo "S7 SHA256SUMS is data, never executed"
EVIL=$RELEASES/dist-evil
rm -rf "$EVIL"; cp -r "$RELEASES/dist2" "$EVIL"
# shellcheck disable=SC2016  # literal command substitutions: they must never run
printf '$(touch %s/pwned)  kokoro-x86_64-linux.tar.gz\n`touch %s/pwned2`\n' "$WORK" "$WORK" > "$EVIL/SHA256SUMS"
MOCKPATH='' run_install s7 "$P" KOKORO_VERSION=v0.1.0-test4 KOKORO_RELEASE_URL="file://$EVIL"
check "exit non-zero" test "$RC" != 0
check "malformed checksum file rejected" has "no valid entry for kokoro-x86_64-linux.tar.gz"
check "nothing executed" bash -c "! ls '$WORK'/pwned* 2>/dev/null"

echo "S8 interrupted model download (TERM during a real Hugging Face download): nothing half-installed"
P8=$WORK/home-interrupt
mkdir -p "$P8"
# slow the download down (curl reads ~/.curlrc of this test HOME) so the signal lands mid-download
printf 'limit-rate = 2M\n' > "$P8/.curlrc"
( cd "$WORK" && env -i HOME="$P8" PATH=/usr/bin:/bin KOKORO_VERSION=$V1 KOKORO_RELEASE_URL="$URL1" \
  timeout -s TERM 12 bash "$REPO/install.sh" > "$WORK/s8.log" 2>&1 ); RC=$?; OUT=$(cat "$WORK/s8.log")
echo "[s8] exit=$RC"
D8=$P8/.local/share/kokoro
check "interrupted (non-zero exit)" test "$RC" != 0
check "no launcher" no_file "$P8/.local/bin/kokoro"
check "no partial weights file" no_file "$D8/models/$REV/kokoro-v1_0.pth"
check "staging removed" no_staging "$D8"
rm -f "$P8/.curlrc"
MOCKPATH='' run_install s8b "$P8" KOKORO_VERSION=$V1 KOKORO_RELEASE_URL="$URL1" KOKORO_MODEL_URL="$MIRROR_URL"
check "rerun completes" test "$RC" = 0

echo "S9 corrupt model download"
P9=$WORK/home-badmodel
BADM=$WORK/model-bad
cp -r "$MIRROR" "$BADM"
printf 'X' | dd of="$BADM/voices/am_adam.pt" bs=1 seek=100 conv=notrunc status=none
MOCKPATH='' run_install s9 "$P9" KOKORO_VERSION=$V1 KOKORO_RELEASE_URL="$URL1" KOKORO_MODEL_URL="file://$BADM"
check "exit non-zero" test "$RC" != 0
check "model checksum mismatch reported" has "checksum mismatch for model file voices/am_adam.pt"
check "corrupt voice not installed" no_file "$P9/.local/share/kokoro/models/$REV/voices/am_adam.pt"
check "no launcher" no_file "$P9/.local/bin/kokoro"

# --- eSpeak NG consent (MOCK ldconfig hides libespeak-ng.so.1; MOCK sudo/apt-get) ---------------
echo "S10 eSpeak NG missing, no terminal (KOKORO_TTY unusable)"
P10=$WORK/home-notty
MOCKPATH=$MOCKS/std run_install s10 "$P10" MOCK_HIDE_ESPEAK=1 KOKORO_TTY=/nonexistent/tty KOKORO_VERSION=$V1 KOKORO_RELEASE_URL="$URL1" KOKORO_MODEL_URL="$MIRROR_URL"
check "exit non-zero" test "$RC" != 0
check "manual command shown" has "sudo apt-get install libespeak-ng1"
check "sudo never called" no_file "$WORK/state/s10/sudo.log"
check "nothing installed" no_file "$P10/.local/share/kokoro"

echo "S11 eSpeak NG missing, no controlling terminal at all (real /dev/tty, setsid)"
P11=$WORK/home-setsid
mkdir -p "$P11" "$WORK/state/s11"
curl -fsSL "file://$REPO/install.sh" | setsid -w env -i HOME="$P11" PATH="$MOCKS/std:/usr/bin:/bin" MOCK_STATE="$WORK/state/s11" \
  MOCK_HIDE_ESPEAK=1 KOKORO_VERSION=$V1 KOKORO_RELEASE_URL="$URL1" bash > "$WORK/s11.log" 2>&1
RC=${PIPESTATUS[1]}; OUT=$(cat "$WORK/s11.log"); echo "[s11] exit=$RC"
check "exit non-zero" test "$RC" != 0
check "no-terminal message" has "no terminal to ask for permission"
check "sudo never called" no_file "$WORK/state/s11/sudo.log"

echo "S12 eSpeak NG missing, user declines"
for ans in n ""; do
  printf '%s\n' "$ans" > "$WORK/tty-$ans-answer"
  P12=$WORK/home-decline-$ans
  MOCKPATH=$MOCKS/std run_install "s12$ans" "$P12" MOCK_HIDE_ESPEAK=1 KOKORO_TTY="$WORK/tty-$ans-answer" KOKORO_VERSION=$V1 KOKORO_RELEASE_URL="$URL1" KOKORO_MODEL_URL="$MIRROR_URL"
  check "answer '$ans': exit non-zero" test "$RC" != 0
  check "answer '$ans': question asked" has "[y/N]"
  check "answer '$ans': sudo never called" no_file "$WORK/state/s12$ans/sudo.log"
  check "answer '$ans': nothing installed" no_file "$P12/.local/share/kokoro"
done

echo "S13 eSpeak NG missing, user agrees; XDG_DATA_HOME and KOKORO_INSTALL_DIR respected"
printf 'y\n' > "$WORK/tty-yes"
P13=$WORK/home-consent
MOCKPATH=$MOCKS/std run_install s13 "$P13" MOCK_HIDE_ESPEAK=1 KOKORO_TTY="$WORK/tty-yes" XDG_DATA_HOME="$P13/xdg" KOKORO_INSTALL_DIR="$P13/mybin" \
  KOKORO_VERSION=$V1 KOKORO_RELEASE_URL="$URL1" KOKORO_MODEL_URL="$MIRROR_URL"
check "exit 0" test "$RC" = 0
check "sudo apt-get install -y libespeak-ng1 called once" test "$(cat "$WORK/state/s13/sudo.log" 2>/dev/null)" = "sudo apt-get install -y libespeak-ng1"
check "apt-get stdin is the terminal, not the piped script" grep -qF "stdin=$WORK/tty-yes" "$WORK/state/s13/apt.log"
check "data under XDG_DATA_HOME" test -d "$P13/xdg/kokoro/releases/$V1"
check "command in KOKORO_INSTALL_DIR" test -x "$P13/mybin/kokoro"
check "nothing in ~/.local" no_file "$P13/.local"

echo "S14 user agrees, apt-get fails"
P14=$WORK/home-aptfail
MOCKPATH=$MOCKS/std run_install s14 "$P14" MOCK_HIDE_ESPEAK=1 MOCK_APT_EXIT=100 KOKORO_TTY="$WORK/tty-yes" KOKORO_VERSION=$V1 KOKORO_RELEASE_URL="$URL1"
check "exit non-zero" test "$RC" != 0
check "failure reported" has "sudo apt-get install -y libespeak-ng1' failed"
check "nothing installed" no_file "$P14/.local/share/kokoro"

# --- other preconditions ----------------------------------------------------------------------
echo "S15 cuBLAS missing (MOCK ldconfig hides libcublas.so.12)"
P15=$WORK/home-nocublas
MOCKPATH=$MOCKS/std run_install s15 "$P15" MOCK_HIDE_CUBLAS=1 KOKORO_VERSION=$V1 KOKORO_RELEASE_URL="$URL1"
check "exit non-zero" test "$RC" != 0
check "names the missing library" has "libcublas.so.12 (cuBLAS, CUDA 12)"
check "says it does not install CUDA" has "does not install them"
check "nothing installed" no_file "$P15/.local/share/kokoro"

echo "S16 unsupported machine (MOCK uname: aarch64)"
ln -sf "$MOCKS/uname" "$MOCKS/no-uname/uname"
P16=$WORK/home-arm
MOCKPATH=$MOCKS/no-uname run_install s16 "$P16" KOKORO_VERSION=$V1 KOKORO_RELEASE_URL="$URL1"
check "exit non-zero" test "$RC" != 0
check "platform message" has "Linux x86_64 only"

echo "S17 an unrelated ~/.local/bin/kokoro is never replaced"
P17=$WORK/home-foreign
mkdir -p "$P17/.local/bin"
printf '#!/bin/sh\necho mine\n' > "$P17/.local/bin/kokoro"
cp "$P17/.local/bin/kokoro" "$WORK/foreign.before"
MOCKPATH='' run_install s17 "$P17" KOKORO_VERSION=$V1 KOKORO_RELEASE_URL="$URL1" KOKORO_MODEL_URL="$MIRROR_URL"
check "exit non-zero" test "$RC" != 0
check "refusal message" has "was not written by this installer"
check "file unchanged" cmp -s "$WORK/foreign.before" "$P17/.local/bin/kokoro"
check "nothing downloaded" no_file "$P17/.local/share/kokoro"

echo "S18 KOKORO_RELEASE_URL requires KOKORO_VERSION"
P18=$WORK/home-noversion
MOCKPATH='' run_install s18 "$P18" KOKORO_RELEASE_URL="$URL1"
check "exit non-zero" test "$RC" != 0
check "message" has "KOKORO_RELEASE_URL needs KOKORO_VERSION"

echo "S19 latest-release lookup against the real GitHub repository"
P19=$WORK/home-latest
MOCKPATH='' run_install s19 "$P19"
if [[ $RC == 0 ]]; then
  check "installed the latest published release" test -x "$P19/.local/bin/kokoro"
else
  check "no published release: clear failure" bash -c "grep -qE 'no published release|could not reach' '$WORK/s19.log'"
  check "nothing installed" no_file "$P19/.local/share/kokoro/releases"
fi

echo "S20 explicit paths still override the installed defaults"
env -i HOME="$P" PATH="$P/.local/bin:/usr/bin:/bin" CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}" \
  kokoro synth --model-dir "$WORK/no-such-model" --input "$WORK/s1.txt" --out-dir "$WORK/s20-out" > "$WORK/s20a.log" 2>&1
RC=$?; OUT=$(cat "$WORK/s20a.log")
check "--model-dir override used (exit 2 naming it)" bash -c "[[ $RC == 2 ]] && grep -qF '$WORK/no-such-model' '$WORK/s20a.log'"
env -i HOME="$P" PATH="$P/.local/bin:/usr/bin:/bin" CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}" KOKORO_FRONTEND_DIR="$WORK/no-such-frontend" \
  kokoro synth --input "$WORK/s1.txt" --out-dir "$WORK/s20-out" > "$WORK/s20b.log" 2>&1
RC=$?
check "KOKORO_FRONTEND_DIR override used (exit 2 naming it)" bash -c "[[ $RC == 2 ]] && grep -qF '$WORK/no-such-frontend' '$WORK/s20b.log'"

echo
echo "installer scenarios: $PASSES checks passed, $FAILS failed"
[[ $FAILS == 0 ]]
