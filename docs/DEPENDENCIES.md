# Runtime dependency inventory — `kokoro synth`

Audited 2026-09-26 on the RTX 4090 host with `strace -f -e trace=openat,execve` of
`kokoro synth` (text input, native frontend, CUDA) under an empty environment
(`env -i`, `PATH=/nonexistent`). "One executable" here means that one Rust program does all
inference work, from text to WAV, in one process. It does NOT mean that the model and language
data are embedded, or that the build is static and dependency-free. Everything below is loaded
from explicit paths at runtime, except eSpeak NG, which is the system installation unless overridden.

## What is NOT required at inference
- **No Python** interpreter, runtime, package or bridge process. The `python-bridge` frontend was
  removed from the binary. Tested with Python unavailable in `tests/cli_text_native.rs`.
- **No subprocesses.** The execve audit sees exactly one exec, the binary itself, in every
  integrated test and in the full-chapter acceptance.
- **No network** and no oracle replay: nothing recorded from the reference is read at inference.

## Executable
| item | detail |
|---|---|
| `target/release/kokoro` | Rust, built `cargo build --release` (CUDA is always required; there is no CPU build). `ldd` shows only libc, libm and libgcc_s. The CUDA kernels are compiled to PTX at build time (nvcc 12.9, compute_89) and embedded in the binary (`include_str!`). They are JIT-loaded by the driver. |
| build configuration | One configuration: the BF16x build, kernels compiled with FMA contraction (`-fmad=true`). Recorded in every sidecar as `cuda bf16x kernels=fma(-fmad=true)`. (Earlier f32/FMA and experimental precision builds were removed; see docs/HISTORY.md.) |

## Native libraries loaded at runtime
| library | how | version on this host | license / note |
|---|---|---|---|
| libcuda.so (NVIDIA driver) + libnvidia-ptxjitcompiler | dlopen (cudarc dynamic loading) | driver 580.178.04 | NVIDIA driver. Required. |
| libcublas.so + libcublasLt.so.12 | dlopen | CUDA 12.9 (12.9.2.10), /usr/local/cuda | NVIDIA CUDA toolkit runtime. Required. |
| **libespeak-ng.so.1** (eSpeak NG) + its `espeak-ng-data/` | dlopen of the separately installed system library (default name `libespeak-ng.so.1` via the dynamic loader, or `--espeak-lib`); data from the library's default location (or `--espeak-data`) | 1.51 (Ubuntu package `libespeak-ng1`, which pulls in `espeak-ng-data`) | **GPL-3.0-or-later.** Required for text input: a native C library used in-process for the pronunciation of words missing from the lexicon (not for neural synthesis). It is not included in or distributed with this project, not compiled in, and not run as a subprocess. Any version ≥ 1.49 with the C API (`espeak_Initialize`, `espeak_Info`, `espeak_SetVoiceByName`, `espeak_TextToPhonemes`) is accepted; the version, library and data hashes are recorded. Its own dependencies are libstdc++, libm and libgcc_s. This notice describes the dependency; it is not a license assessment of combining it with this program. |
| libstdc++, libm, libgcc_s, libc, libdl, libpthread, librt | system | Ubuntu | loaded via espeak-ng and the CUDA driver |

Device nodes: /dev/nvidiactl, /dev/nvidia-uvm and /dev/nvidia0 are opened by the driver. It also
probes /dev/nvidia1 during initialisation even with `CUDA_VISIBLE_DEVICES=0`. No context is
created on GPU 1.

## External data (explicit paths; hashed into the run identity, so a change invalidates resume)
| data | path (this host) | size | sha256 | license / provenance |
|---|---|---|---|---|
| Kokoro-82M weights `kokoro-v1_0.pth`, `config.json`, `voices/<voice>.pt` | `--model-dir` (HF snapshot f3ff3571…) | 327 MB + 0.5 MB/voice | docs/truth-pack (weights 496dba11…) | Apache-2.0 (model card) |
| misaki 0.9.4 lexicons `us_gold.json`, `us_silver.json` | `<frontend-dir>/misaki-0.9.4/` | 5.1 MB (with the unused gb_*) | dc414872… / de8f67be… | misaki Apache-2.0. **Lexicon data provenance is an open item** (docs/frontend/COVERAGE.md) |
| spaCy en_core_web_sm 3.8.0: tokenizer rules, lexeme_norm lookups, BASE_NORMS, symbol table, tok2vec + tagger weights (exported by oracle/export_spacy.py, deterministic) | `<frontend-dir>/spacy-en_core_web_sm-3.8.0/` | 6.2 MB | tokenizer.json b44b0bc0…, lookups.json a9162beb…, base_norms.json dfddb78e…, symbols.json 3f7694e0…, model_structure.json f80547c2…, tagger_weights.safetensors 66a133a4… | spaCy + model MIT |
| eSpeak NG en-us data (`phontab`, `phonindex`, `phondata`, `intonations`, `en_dict`, `lang/gmw/en-US`) | the system `espeak-ng-data/` (Ubuntu: /usr/lib/x86_64-linux-gnu/espeak-ng-data) | 0.8 MB of the package's data | hashed at load time (identity) | GPL-3.0-or-later (see above) |

## Optional (off by default)
- `--encode <ext>` runs the external **`ffmpeg`** executable (found on PATH) as a subprocess, to
  also write FLAC/MP3/Opus next to each WAV. The encoded file's sha256 is recorded in the sidecar.
  It is not used by the default WAV path or by the no-Python /
  no-subprocess acceptance runs. `tests/cli_text_native.rs::encode_with_ffmpeg_when_enabled`
  covers it. With `--encode`, the execve audit shows the binary plus one ffmpeg per line.

## Build-time only
Rust toolchain plus crates (Cargo.lock), and the CUDA toolkit nvcc 12.9 for the PTX. The oracle/,
bench/ and scripts/ Python tooling is for development and differential testing only. The
reference venv is used to generate fixtures and to stage its eSpeak NG 1.52.0 copy for the
reference-comparison tests, not at inference.
