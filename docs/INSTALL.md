# Installing kokoro

```
curl -fsSL https://raw.githubusercontent.com/mdenil/kokoro-rust/main/install.sh | bash
```

`install.sh` installs a GitHub release of kokoro for the current user. It needs no root access and
no Python or compiler, only `bash`, `curl`, `tar`, `gzip` and `sha256sum`.

## What it does
1. Checks the platform (Linux x86_64) and the GPU libraries the program loads at run time:
   `libcuda.so.1` and `libnvidia-ptxjitcompiler.so.1` (NVIDIA driver), and `libcublas.so.12` and
   `libcublasLt.so.12` (cuBLAS from CUDA 12). Missing ones are listed and the installer stops. It
   never installs or bundles the NVIDIA driver or CUDA.
2. Checks for the eSpeak NG library (`libespeak-ng.so.1`). If it is missing, the installer asks on
   your terminal whether to run `sudo apt-get install libespeak-ng1` (sudo may ask for your
   password). Without a terminal, or if you answer no, it prints that command and stops. On systems
   without `apt-get`, install the package that provides `libespeak-ng.so.1` yourself.
3. Finds the latest published release and downloads its `SHA256SUMS`, `kokoro-x86_64-linux.tar.gz`
   (the program) and `kokoro-frontend.tar.gz` (language data), and checks both packages against
   the checksums.
4. Downloads the Kokoro-82M model (`config.json`, `kokoro-v1_0.pth`) and the voices `af_heart` and
   `am_adam` from Hugging Face, revision `f3ff3571`, each checked against a SHA-256 built into the
   installer.
5. Writes the `kokoro` command.

Downloads are checked in a temporary directory inside the data directory before anything installed
changes:
- The new release is moved into place only after the release and the model are complete.
- Model files are replaced one at a time, each only after its checksum matches.
- When a damaged or re-published copy of the same version is replaced, the old copy is kept until
  the `kokoro` command has been written, and put back if any step fails or the run is interrupted.
- The temporary directory is removed at the end of every run.
- If the installer is killed outright (for example by `kill -9`) while a release is being moved into
  place, the next run restores the old copy.

## Where things go
| path | contents |
|---|---|
| `~/.local/bin/kokoro` | the command: a small script that starts the installed program with the installed data |
| `~/.local/share/kokoro/releases/<version>/` | the program (`bin/kokoro`), the language data (`frontend/`), license files |
| `~/.local/share/kokoro/models/f3ff3571…/` | model and voices, shared by all installed versions |

`$XDG_DATA_HOME/kokoro` replaces `~/.local/share/kokoro` when `XDG_DATA_HOME` is set. The installer
does not edit shell startup files; if `~/.local/bin` is not on your `PATH`, it says so.

The installed command uses the installed model and language data unless you pass `--model-dir` /
`--frontend-dir` or set `KOKORO_MODEL_DIR` / `KOKORO_FRONTEND_DIR`.

## Options
Environment variables, all optional:

| variable | effect |
|---|---|
| `KOKORO_VERSION` | install this release tag instead of the latest, e.g. `KOKORO_VERSION=v0.1.0` |
| `KOKORO_INSTALL_DIR` | directory for the `kokoro` command (default `~/.local/bin`) |
| `XDG_DATA_HOME` | parent of the data directory (default `~/.local/share`) |
| `KOKORO_RELEASE_URL` | base URL of a mirror holding the release files (requires `KOKORO_VERSION`) |
| `KOKORO_MODEL_URL` | base URL of a mirror of the model files at revision `f3ff3571` |
| `KOKORO_TTY` | device to read the eSpeak NG question from (default `/dev/tty`) |

For example: `curl -fsSL …/install.sh | KOKORO_VERSION=v0.1.0 bash`.

## Updating and removing
- Run the installer again to update to the latest release. The model is downloaded only if it is
  missing or damaged. A release that is already installed and intact is kept as it is; a damaged
  one is replaced.
- Older versions stay in `~/.local/share/kokoro/releases/`; the installer lists them, and they can
  be deleted.
- The installer never replaces a `kokoro` command that it did not write.
- To uninstall: `rm ~/.local/bin/kokoro` and `rm -r ~/.local/share/kokoro`.

## Release packages
| file | contents |
|---|---|
| `kokoro-x86_64-linux.tar.gz` | `kokoro-x86_64-linux/`: `bin/kokoro`, `VERSION`, `LICENSE` (Apache-2.0), `README.md`, `THIRD_PARTY.md`, `licenses/crates/` (license files of the Rust crates used to build the program) and `licenses/CRATES.tsv` |
| `kokoro-frontend.tar.gz` | `frontend/`: `misaki-0.9.4/` (lexicons), `spacy-en_core_web_sm-3.8.0/` (exported tokenizer and tagger data), `VERSION`, `LICENSE`, `THIRD_PARTY.md`, `licenses/` (misaki, spaCy, en_core_web_sm) |
| `SHA256SUMS` | SHA-256 of the two packages |

The model is not part of the release; the installer downloads it from Hugging Face. eSpeak NG,
the NVIDIA driver and cuBLAS are not included either.

## Making a release
`.github/workflows/release.yml` builds the packages on GitHub:
- the program is compiled in an Ubuntu 22.04 container with CUDA 12.9 `nvcc` and Rust 1.98.1;
- the frontend data is exported with Python on Ubuntu 24.04;
- `scripts/package_release.sh` packages both and writes `SHA256SUMS`;
- a smoke job checks the packages and starts the program on a plain Ubuntu 22.04 (no GPU there, so
  it doesn't synthesize).

To release, push a tag starting with `v` (it can point to any commit):
```
git tag v0.1.0 && git push origin v0.1.0
```
The workflow then creates a draft GitHub release with the three files. Check it, and publish it by
hand; the installer only sees published releases.

To test the build without releasing, start the workflow by hand ("Run workflow" on the Actions
page, or `gh workflow run release.yml --ref <branch>`). It runs the same build, package and smoke
jobs, keeps the packages as downloadable artifacts of that run (`release-packages`), and creates no
release. To install such a package, unpack the downloaded artifact into a directory and point the
installer at it: `KOKORO_VERSION=v0.0.0-build.<run number> KOKORO_RELEASE_URL=file:///that/dir bash install.sh`.

The same script builds the packages locally:
```
scripts/package_release.sh binary v0.1.0 dist
scripts/fetch_assets.sh --frontend-only data
scripts/prepare_spacy_assets.sh data
scripts/package_release.sh frontend v0.1.0 dist data
scripts/package_release.sh sums dist
```
`tests/installer/test_install.sh` exercises the installer against locally built packages.
