# Portability and configuration

## Supported scope (initial)
| | tested | notes |
|---|---|---|
| OS / CPU | Linux x86_64 (Ubuntu) | other platforms unverified |
| GPU | NVIDIA RTX 4090, compute capability 8.9 | the kernels are PTX for compute_89. Older devices are refused at startup; newer ones may JIT the PTX but are **unverified**, and a warning is printed |
| CUDA | toolkit 12.9 (nvcc at build, cuBLAS at run time), driver 580 | |
| Numerics | the accepted BF16x configuration only (strict `-fmad=false`) | fixed; not configurable |
| Language | American English (misaki 0.9.4 + spaCy en_core_web_sm 3.8.0 + espeak-ng 1.52.0 fallback) | |

Nothing here extends that scope; untested GPUs, drivers and platforms remain unverified.

## Configuration (all explicit; no host-specific defaults)
### Build
- `cargo build --release`. Output goes to cargo's default `target/` of the checkout (gitignored).
  An inherited `CARGO_TARGET_DIR` is honored by cargo as usual; nothing in this repo sets one.
- nvcc lookup order:
  1. `$NVCC`;
  2. `$CUDA_HOME/bin/nvcc`;
  3. `$CUDA_PATH/bin/nvcc`;
  4. `nvcc` on `PATH`;
  5. `/usr/local/cuda/bin/nvcc` (the toolkit's default prefix).
  A missing nvcc is a build error that says so. The compile flags are fixed
  (`-arch=compute_89 -fmad=false -O3`).
- Local, untracked build settings (for example `jobs`) may go in `.cargo/config.toml`, which is
  gitignored. Public commands never depend on it.

### Run (`kokoro synth` / `kokoro bench`)
| setting | option | environment | default |
|---|---|---|---|
| model snapshot (config.json, kokoro-v1_0.pth, voices/) | `--model-dir` | `KOKORO_MODEL_DIR` | none (required) |
| frontend data (misaki-0.9.4/, spacy-en_core_web_sm-3.8.0/, espeak-ng-1.52.0/) | `--frontend-dir` | `KOKORO_FRONTEND_DIR` | none (required for text input) |
| libespeak-ng 1.52.0 | `--espeak-lib` | | `<frontend-dir>/espeak-ng-1.52.0/libespeak-ng.so.1.52.0` |
| GPU | `--cuda-device N` (index among visible devices) | `CUDA_VISIBLE_DEVICES` (driver) | device 0 of the visible set |
| ffmpeg (only with `--encode`) | | `PATH` | |

- Missing or incomplete model/frontend directories, an unavailable or unsupported GPU, and missing
  input files fail with exit 2 and a message that names the option to fix.
- Relative paths resolve against the working directory.
- The binary reads no other environment variables, except two:
  - `KOKORO_PRECISION`, only to refuse values other than `bf16x`;
  - the test fault hook `KOKORO_BATCH_NEGCTL_NOMASK`.
- No Python and no network at run time.

### Tests
| variable | meaning |
|---|---|
| `KOKORO_DATA` (required) | test data root: `fixtures/`, `frontend/`, `hf/` (pinned model snapshot), `models/`; the optional private tests also need `KOKORO_PRIVATE_CHAPTER` and reference dumps under `evidence/private/`. Unset = the tests fail with a message, never a vacuous pass |
| `KOKORO_MODEL_DIR`, `KOKORO_FRONTEND_DIR` | optional overrides of the data-root defaults |
| `CUDA_VISIBLE_DEVICES` | passed through unchanged to the binary under test; never forced |
| `KOKORO_BIN` | optional binary under test (default: this crate's) |
| `KOKORO_ESPEAK_NEGCTL_LIB` | optional non-pinned libespeak-ng for the refusal control (default: found in the standard system library dirs) |

- Scratch output goes to cargo's per-project `CARGO_TARGET_TMPDIR` (inside `target/`).
- Host tools the tests need:
  - `strace` on PATH (execve audit);
  - `ffmpeg` on PATH (`--encode` test);
  - a non-1.52 libespeak-ng (refusal negative control).
  Each of these fails with an actionable message when absent.
- A data root can be an "isolated view" of symlinks to the pinned asset directories, as used for the
  check below.

### Developer scripts (not needed by the public commands)
- `scripts/env.sh` derives the checkout root from its own location.
- `KOKORO_DATA` comes from the caller or from the untracked `scripts/env.local.sh`; otherwise it
  refuses to continue.
- It exports tool caches under the data root and unsets `CARGO_TARGET_DIR`.
- The Python oracle/bench tools require `KOKORO_DATA` (actionable exit otherwise) and inherit the
  caller's `CUDA_VISIBLE_DEVICES`.
- `scripts/setup_reference_env.sh` takes `$REFERENCE_PYTHON` (CPython 3.12.3; default `python3.12`).
- `scripts/check_private_leaks.py` refuses to report "clean" without the private corpus it audits
  against.

## Checking a setup
The configuration contract above is exercised by the test suite under a stripped environment:

```
env -i HOME=$HOME PATH=<cargo bin dir>:/usr/bin:/bin \
    KOKORO_DATA=/path/to/data CUDA_VISIBLE_DEVICES=0 \
    cargo test --release -- --test-threads=2
```

- `KOKORO_DATA` may be a directory of symlinks to the pinned asset directories (`fixtures/`,
  `frontend/`, `hf/`, `models/`); the public tests need nothing else.
- With `KOKORO_DATA` unset the tests fail with an explanatory message instead of passing
  vacuously.
- The product's own configuration errors (missing model or frontend data, no usable GPU, obsolete
  `KOKORO_PRECISION`) exit with status 2 and name the option to fix.
