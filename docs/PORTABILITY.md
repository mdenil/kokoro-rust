# Portability and configuration

## Supported scope (initial)
| | tested | notes |
|---|---|---|
| OS / CPU | Linux x86_64 (Ubuntu) | other platforms unverified |
| GPU | NVIDIA RTX 4090, compute capability 8.9 | the kernels are PTX for compute_89. Older devices are refused at startup; newer ones may JIT the PTX but are **unverified**, and a warning is printed |
| CUDA | toolkit 12.9 (nvcc at build, cuBLAS at run time), driver 580 | |
| Numerics | the owner-accepted BF16x configuration only (strict `-fmad=false`) | fixed; not configurable |
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
- Local, untracked build settings may go in `.cargo/config.toml` (gitignored). This host uses one
  for `jobs = 8` (build parallelism only). Public commands never depend on it.

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
| `KOKORO_DATA` (required) | test data root: `fixtures/`, `frontend/`, `hf/` (pinned model snapshot), `models/`; private tests also need `bench/private/` and `evidence/private/`. Unset = the tests fail with a message, never a vacuous pass |
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
  milestone check below.

### Developer scripts (not needed by the public commands)
- `scripts/env.sh` derives the checkout root from its own location.
- `KOKORO_DATA` comes from the caller or from the untracked `scripts/env.local.sh`; otherwise it
  refuses to continue.
- It exports tool caches under the data root and unsets `CARGO_TARGET_DIR`.
- The Python oracle/bench tools require `KOKORO_DATA` (actionable exit otherwise) and inherit the
  caller's `CUDA_VISIBLE_DEVICES`.
- `scripts/ab.sh` needs `AB_PAIR_SCRIPT` (an external runner).
- `scripts/setup_reference_env.sh` takes `$REFERENCE_PYTHON` (CPython 3.12.3; default `python3.12`).
- `scripts/check_private_leaks.py` refuses to report "clean" without the private corpus it audits
  against.

## Portability inventory (milestone owner #26 follow-up; worker-independent assumptions removed)
| was | now |
|---|---|
| product: espeak default dir `$KOKORO_DATA` or `/data/mdenil/...` | removed from the product; tests derive it from `KOKORO_FRONTEND_DIR` / `KOKORO_DATA` |
| tests: `KOKORO_DATA` default `/data/mdenil/code/kokoro-rust` (12 files); scratch under the data root | `tests/support/paths.rs`: required `KOKORO_DATA`, optional overrides, scratch in `CARGO_TARGET_TMPDIR` |
| tests: `CUDA_VISIBLE_DEVICES=0` forced into the binary under test | caller's value passed through |
| tests: `/usr/bin/strace`, ffmpeg from `/usr/bin` | resolved on PATH, actionable failure |
| tests: `/usr/lib/x86_64-linux-gnu/libespeak-ng.so.1.1.51` | `KOKORO_ESPEAK_NEGCTL_LIB` or standard library dirs |
| build.rs: nvcc only `$NVCC` or `/usr/local/cuda/bin/nvcc` | `$NVCC`, `$CUDA_HOME`, `$CUDA_PATH`, PATH, default prefix; actionable error |
| `.cargo/config.toml` (committed `jobs = 8`) | untracked local setting (gitignored) |
| `scripts/env.sh`: `/home/mdenil/...`, `/data/mdenil/...`, `CARGO_TARGET_DIR` into another checkout | derived root, explicit/local data root, cargo default target |
| `scripts/ab.sh`: runner under `/home/mdenil/.claude/...` | `AB_PAIR_SCRIPT` |
| `scripts/setup_reference_env.sh`: `/usr/bin/python3` | `$REFERENCE_PYTHON` |
| oracle/bench Python: `KOKORO_DATA` default `/data/mdenil/...`; `CUDA_VISIBLE_DEVICES=0` forced; retired-root guard | required `KOKORO_DATA`; caller's GPU selection; guard removed |
| leak checker: silent default data root | required; refuses to audit against nothing |
| `tests/pinned/bf16x_golden.json` provenance path | `$KOKORO_DATA/bin/...` (hashes unchanged) |
| `tests/pinned/regression_bounds.json` (f32 RB-1 bounds, `/data/mdenil` path) | removed; no remaining user |

Credentials audit: the 126 tracked files were scanned for private-key blocks, GitHub /
Hugging Face / AWS / Slack / API-key token formats, assigned secrets and credentials embedded in
URLs. No findings; no values were printed.

## Verification (tree 3895666 = portability work on top of canonical main fc30599)
Evidence:
- `$KOKORO_DATA/evidence/release-cleanup/portability-3895666/`: per step `<step>.cmd` (exact
  command), `<step>.log` (full output) and `<step>.exit`, plus SHA256SUMS.
- Private-suite logs: `$KOKORO_DATA/evidence/private/release-cleanup/portability-3895666/`.

Every step runs under `env -i` with only HOME, a minimal PATH and the variables shown.

| step | configuration | result |
|---|---|---|
| cargo metadata | no CARGO_TARGET_DIR | target_directory = the checkout's `./target` |
| default suite | `KOKORO_DATA=<isolated view>` (symlinks to fixtures/, frontend/, hf/, models/ only), `CUDA_VISIBLE_DEVICES=0` | exit 0: 40 passed, 0 failed, 9 ignored. Includes the public goldens (byte-identical to the accepted artifact), the leakage negative control, the Python-free execve audit and the refusal of a non-pinned espeak |
| no data root | `KOKORO_DATA` unset | exit 101, "KOKORO_DATA is not set ... NOT a pass" |
| ignored suites: golden (private chapter), reference diagnostic, frontend spacy/espeak/g2p/pipeline, CLI private chapter | real data root | all exit 0; private chapter 316/316 lines byte-identical, both voices |
| missing `--model-dir` / incomplete model dir / missing or incomplete frontend dir / no visible GPU / invalid `--cuda-device` / obsolete `KOKORO_PRECISION` | explicit options | exit 2, each with a message naming what to fix |
| relative input + output paths from another working directory, model/frontend from `KOKORO_MODEL_DIR` / `KOKORO_FRONTEND_DIR` | | exit 0, complete output |
| `scripts/env.sh`, leak checker, golden generator without `KOKORO_DATA` | | exit 1, actionable message |
| build with `NVCC=/nonexistent/nvcc` | | fails: "NVCC=... does not exist. Install the CUDA 12.9 toolkit or set NVCC=..." |
| build with no nvcc hints (minimal PATH) | | exit 0, nvcc found at the default prefix |

GPU: all runs used this host's authorized GPU 0 via `CUDA_VISIBLE_DEVICES=0`, set by the caller; no
test forces it.

## Deferred to a later milestone (inventory only)
Host-specific paths remain in historical records and briefs; they are not product
instructions. Occurrences of `/home/mdenil`, `/data/mdenil`, `CUDA_VISIBLE_DEVICES=0` or
`--features cuda`:
- HERMES_BRIEF.md (11), PORT_STATE.md (4), the other HERMES_*_BRIEF.md files (1 each);
- docs/PERF_LEDGER.md (14), docs/conformance/TEST_RECEIPTS.md (8), docs/PERFORMANCE_REPORT.md (6),
  docs/RELOCATION_2026-09-26.md (5), docs/truth-pack/PINNED_SOURCES.md (4),
  docs/conformance/NONDET_FLOOR.md (2), docs/conformance/REGRESSION_POLICY.md (2);
- one each in docs/conformance/TOLERANCE_HISTORY.md, docs/NEGATIVE_EVIDENCE.md,
  docs/PHASE2_PRECISION.md, docs/truth-pack/OQ_INDEX.md, docs/RELEASE_CLEANUP.md (the
  "previously: --features cuda" note).
Clean-checkout setup validation and license/provenance decisions are also later gates.
