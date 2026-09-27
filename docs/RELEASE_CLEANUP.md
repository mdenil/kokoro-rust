# BF16x-only release cleanup (owner #26) — inventory and evidence

Branch `release/bf16x-cleanup`, from `aed6a5e`: the accepted code tree 9b39d48 plus docs. Scope:
HERMES_RELEASE_CLEANUP_BRIEF.md.

## Reference artifact (immutable)
- Binary: `$KOKORO_DATA/bin/phase2-9b39d48-6fde9d88990a`, sha256
  6fde9d88990a8dc518ec1f366fb17db0869f7e7df5fc1fdea973ea0a371612ea.
- Run as: `KOKORO_PRECISION=bf16x`, every other option at its default (owner #25 acceptance).
- Frozen numerical configuration (read from the accepted build and its defaults):
  - Kernels built with nvcc `-fmad=false` (strict rounding; NOT the earlier FMA option),
    `-arch=compute_89`.
  - BF16 operands with round-to-nearest-even conversion, f32 accumulate and output:
    - every stride-1 convolution of the text encoder, predictor, decoder and generator (fused
      tensor-core kernel when Cout % 64 = 0, Cin % 16 = 0 and the output is same-length; per-tap
      cuBLAS otherwise);
    - every linear layer (ALBERT, LSTM input projections, AdaIN style projections, encoders).
  - Phase-2 levers active: P2-L2 (v2 fused kernel), P2-L3 (128-channel blocks), P2-L4 (AdaIN + Snake
    prologue in the Snake blocks), P2-L5 (float-partial channel statistics).
  - f32: the two strided convs (conv1d_strided), the 1x1 Cin-22 conv (tiled direct kernel), LSTM
    recurrences (persistent cooperative kernel), attention products (cuBLAS SGEMM), transposed
    convs, normalization, source/STFT/iSTFT.
  - Batched synthesis: 8000 phoneme chars / 64 items per microbatch, window 256 (first 32),
    failing batches split down to single items.

## Regression goldens (established BEFORE any code change)
- `bench/make_bf16x_golden.py` renders each case twice with the accepted binary: every case is
  deterministic.
- Hashes: `tests/pinned/bf16x_golden.json`.
- Cases (11 public): Alice ch. 1 ×2 voices, frontend edge cases ×2, link features, held-out ×2,
  default pcm16 output, speed 0.8, seed 7, one item per batch (`--batch-items 1`).
- Private chapter (316 lines × 2 voices): hashes only under `$KOKORO_DATA/evidence/private/release-cleanup/golden`.
- Test: `tests/bf16x_golden.rs`. Byte equality per line, including line coverage; a negative-control
  test for the checker; the private chapter is an ignored test.
- The test passed on the unmodified accepted code (commit 6968945) before any change.

## Inventory: what the accepted BF16x build actually executes (measured)
Evidence: `$KOKORO_DATA/evidence/release-cleanup/inventory/`.
- **Kernels:** nsys traces of the accepted binary, batched and batch-1 × both voices on 153 public
  lines, plus `bench`. 45 of the 74 custom kernels ran.
  - Never launched by BF16x (29): the FP16 / INT8 / v1-WMMA / CW=2 variants, the f32 fused convs
    (conv1d_igemm, the sliding-window family, the _res variants), chan_stats_seg / chan_stats_seg1,
    conv_direct, lstm_step, lp_absmax*, lp_dequant, lp_transpose_q8/f16, lp_cvt_f16.
- **Conv routes:** a temporary route trace (not committed): routes-batched.txt, routes-batch1.txt.
  Every convolution takes one of 4 routes, fixed by its shape:
  - strided f32: 2 layers;
  - direct tiled f32: the 1x1 Cin-22 conv;
  - fused BF16 (+ prologue in the Snake blocks);
  - per-tap BF16: Cin 514/1090, Cout 22/1.
  The f32 per-tap / fused / sliding-window conv routes never run.

## Switch and option inventory
| switch / option | kind | BF16x value | disposition |
|---|---|---|---|
| `--precision` / `KOKORO_PRECISION` | runtime, numerical mode | bf16x | REMOVED (step A). The flag is unknown to the CLI; the env var is refused unless it equals `bf16x` |
| `KOKORO_FMA` | build, rounding | off (strict) | REMOVED (A); build.rs always compiles `-fmad=false` |
| `KOKORO_IG_BK` | build, f32 fused-conv chunk | 4 | REMOVED (A); constant 4 in the strided kernel |
| `KOKORO_WMMA_TN` | build, v1 WMMA tile | (v1 unused) | REMOVED (A) with the v1 kernels |
| `KOKORO_LP_WMMA`, `KOKORO_LP_WMMA2`, `KOKORO_LP_WMMA2W`, `KOKORO_LP_PROLOGUE` | runtime kill switches (P2-L2..L4) | on | REMOVED (A); the accepted kernels are the only ones |
| `KOKORO_LP_WMMA_ANY` | runtime opt-in (neutral lever) | off | REMOVED (A) |
| `KOKORO_LP_STATS_F32` | runtime kill switch (P2-L5) | on | REMOVED (A); float-partial statistics are the only statistics kernel |
| `KOKORO_STATS_1PASS` | runtime kill switch (PL-014, f32) | not reached in BF16x | REMOVED (A) with the f32 statistics kernels |
| `KOKORO_CONV_IGEMM`, `KOKORO_CONV_IGEMM_MAX_CIN`, `KOKORO_CONV_SW`, `KOKORO_FUSE_RES_CONV` | f32 fused-conv switches (PL-008/009/016) | not reached in BF16x | REMOVED (A) with those kernels |
| `KOKORO_CONV_IGEMM_STRIDED`, `KOKORO_CONV_TILED` | kill switches (PL-012 / PL-004) | on | REMOVED (A); those kernels are the only route of their layers |
| `KOKORO_LSTM_PERSISTENT` | kill switch (PL-002) | on | REMOVED (A) with lstm_step |
| `KOKORO_GPU_HOST_NOISE` | test hook (host noise, batch-1) | off | REMOVED (A) |
| `KOKORO_PREFETCH_FRONTEND`, `KOKORO_SKIP_TEARDOWN` | kill switches (PL-015 / PL-013) | on | REMOVED (A); behavior kept |

| `--device cpu|cuda` | runtime backend selector | cuda | REMOVED (B); CUDA is the only backend. The `cuda` cargo feature is gone and `cudarc` is required |
| `--threads` (CPU math threads; `rayon`, `matrixmultiply`) | CPU backend option | n/a | REMOVED (B) with the CPU backend and both dependencies |
| `--batch-phonemes 0` (batch-1 single-item forward) | alternate GPU implementation | not used (8000) | REMOVED (B): values < 1 are refused. OOM safety is unchanged: failing batches still split down to one item through the same batched path, and `--batch-items 1` runs one chunk per batch |
| `KOKORO_PROFILE` (synchronizing stage profiler, `src/prof.rs`) | diagnostic plumbing | off | REMOVED (B). `--timeline` (non-intrusive span log) stays |
| `KOKORO_DEBUG_F0_DIR` | diagnostic dump | unset | REMOVED (B) |
| `KOKORO_BATCH_NEGCTL_NOMASK` | test fault hook (disables batch gap masking) | unset | KEPT: used by the golden negative control `golden_detects_cross_item_leakage` |

## Backend / implementation inventory
| implementation | disposition |
|---|---|
| CPU f32 model forward (`nn`/`model`/`albert`/`vocoder` forwards, `ops` CPU kernels incl. the `unsafe` matrixmultiply facade, CPU STFT/iSTFT/SineGen) | REMOVED (B). Kept: layer weight structs + loaders (weight-norm resolution), ALBERT host embedding lookup + LayerNorm, duration rounding, noise seeding, constants. `ops` no longer contains `unsafe` |
| GPU single-item forward (`forward_ids`, per-layer `fwd`, `seam_*` oracle entry points) | REMOVED (B), with its 9 kernels (adain_apply, cat_style_rows, chan_stats, expand_rows/cols, istft_frames, lstm_seq, reflect_pad_left1, stft20) |
| GPU batched forward (`forward_batch`) | the product |
| `ItemNoise::Fixed` (replayed reference noise) | KEPT: feeds the reference-fixture diagnostic |

Kernels after cleanup: 36, down from 74. Every one is launched by the product (the Step-A set of 45
minus the 9 single-item kernels).

## Tests
| test | disposition |
|---|---|
| `bf16x_golden` (new) | byte equality vs the accepted artifact: 11 public cases + private chapter (ignored), checker negative controls, cross-item-leakage negative control |
| `bf16x_reference` (new) | exact data pins vs the 15 Python fixtures (input ids, style vectors); ignored diagnostic: product forward with the reference's noise vs the reference audio (printed, no thresholds) |
| `cli_text_native` | kept; `--device` dropped. `batched_and_batch1_agree_on_structure` replaced by `one_item_per_batch_is_complete_and_mapped`: under BF16x, batch shape changes predicted durations (11/65 edge lines), so sample counts / correlation across compositions are not an invariant; both compositions are pinned by the golden test. `long_lines_are_complete` uses one chunk per batch in both runs |
| `cli_linefile` | kept; now runs on the GPU product (was `--device cpu`) |
| frontend suites, `hashing`, `native_load` | unchanged |
| `parity` (CPU f32 ladder, attribution, truth distance, perturbation) | RETIRED with the CPU backend |
| `gpu_parity` (f32 single-item seam ladder, CPU/GPU noise stream, seam negative controls, RB-1 vs strict baseline) | RETIRED with the f32 path and the seam entry points |
| `gpu_batch` (batched vs single-item RB-1 bounds, batching/mapping negative controls, owner listening exports) | RETIRED with the single-item path; mapping is covered by per-line byte equality, leakage by the new negative control |
| `diag_source` (CPU source diagnostic) | RETIRED |
| example `listening_pair` (CPU model) | RETIRED; `load_breakdown` updated |

Historical outcomes of the retired f32 tests stay in the evidence (docs/conformance, `$KOKORO_DATA/evidence/ladder`, phase-1 receipts); they describe the f32 engine, not the product.

## Commits
- 781b910: scope persisted (brief/state).
- 6968945: goldens and golden test, passing on the unmodified accepted code.
- Step A: precision matrix, alternative kernels and kill switches removed. Golden: 0 differences on
  all 11 public cases and the private chapter (both voices).
- Step B: CPU backend, single-item GPU forward, profiling/debug plumbing and the `cuda` feature
  removed; tests repointed/retired. Golden: 0 differences (11 public cases + private chapter);
  leakage negative control detects 65/65 lines.
- Step C (d83540a): product surface: README, CLI description, DEPENDENCIES; bench harness
  engines py | rust | accepted; phase-2 research scripts retired.

## Size
Engine code (src/ without the frontend, kernels, build.rs): 7363 -> 4593 lines (-3125/+349 in
src/kernels/build/Cargo). `unsafe` is confined to the CUDA module (launches, fully-overwritten
allocations, bounds-checked cuBLAS calls); the CPU `ops` facade and its `unsafe` are gone.

## Validation of the cleaned build (tree d83540a)
- Binary `$KOKORO_DATA/bin/bf16x-cleanup-d83540a-a8d405f1cacd` (sha256 a8d405f1…7ddf).
- Evidence: `/data/mdenil/code/kokoro-rust/evidence/release-cleanup/validation-d83540a` (public logs, suite-inventory.txt) and
  `$KOKORO_DATA/evidence/private/release-cleanup/validation-d83540a` (ignored/private suite logs).
- **Default suite** (`cargo test --release --no-fail-fast -- --test-threads=2`): exit 0, 16 targets,
  all pass. Full stdout: suite-default.log.
- **Ignored suites**, each exit 0:
  - bf16x_golden private chapter;
  - bf16x_reference diagnostic;
  - cli_text_native private chapter acceptance (316 lines, pronunciations = reference, 1 exec);
  - frontend_spacy / espeak / g2p / pipeline (private + fuzz/soup corpora).
- **Outputs vs the accepted artifact:** byte-identical on all 11 public golden cases, both voices
  (Alice ch. 1, frontend edge cases, link features, held-out passages, default pcm16, speed 0.8,
  seed 7, one item per batch) and on the full private chapter, both voices. Checked after steps A,
  B and C.
- **Reference data pins:** input ids and voice style vectors equal the 15 Python fixtures exactly.
- **Negative controls:** the golden checker detects changed / missing / extra lines. Disabling
  batch gap masking changes 65/65 edge-case lines and is detected.

## Bounded pre/post throughput smoke (not an optimization campaign)
- Same harness, accepted artifact vs cleaned binary interleaved (3 cold processes, 2 x 3 warm
  passes each). Host load1 5.0–20.8.
- Evidence: `$KOKORO_DATA/evidence/release-cleanup/20260927-095939-smoke-alice` and
  `$KOKORO_DATA/evidence/private/release-cleanup/20260927-095939-smoke-chapter` (aggregates only).

| workload | accepted: cold / warm | cleaned: cold / warm |
|---|---|---|
| Alice ch. 1 (65 lines, 656 s audio) | 2.63 s (cv 2.4%) / 0.99 s (cv 1.9%) | 2.53 s (cv 5.1%) / 1.00 s (cv 1.7%) |
| private chapter (316 lines, 2915 s audio) | 5.63 s (cv 5.2%) / 4.20 s (cv 1.0%) | 5.47 s (cv 2.3%) / 4.14 s (cv 1.7%) |

- No throughput regression: warm within 1.5% either way.
- Model load is ~0.2 s shorter (1.40 -> 1.24 s Alice, 1.43 -> 1.16 s chapter): f32 copies of the
  BF16 weights are no longer uploaded, and BF16 linear weights are converted once at load.
- Every pass of both engines wrote identical WAVs (digest of per-line WAV hashes equal across all
  runs; default pcm16 output).

## Simplified command
```
cargo build --release
kokoro synth --model-dir <snapshot> --frontend-dir <frontend-data> --input book.txt --out-dir out/ \
    [--voice af_heart] [--speed 1.0]
```
Previously: `--features cuda` at build time plus `KOKORO_PRECISION=bf16x` / `--precision bf16x` at run time.

## Remaining packaging / license / install questions (owner decisions; nothing done here)
1. **License.** No project license has been chosen. Cargo.toml has carried
   `license = "Apache-2.0"` since the early scaffolding; it is not an owner decision and should be
   confirmed or removed. libespeak-ng (GPL-3.0-or-later, dlopened) and the model/voice weights have
   their own terms: see docs/DEPENDENCIES.md.
2. **Merge.** Should release/bf16x-cleanup merge into main (main is still the phase-1 f32 engine)?
   No merge was made.
3. **Distribution of data.** The model snapshot and the frontend data (misaki, spaCy, espeak-ng
   1.52 with its library) are loaded from directories: bundle, fetch-script, or document?
4. **GPU scope.** The PTX targets compute_89 (Ada, RTX 40xx), and the BF16 tensor-core path needs
   compute_80+. Is support for other GPUs wanted (a fatbin with more targets) or out of scope?
5. **Install location / packaging.** Binary install path, a service or wrapper for the audiobook
   pipeline, and whether `kokoro bench` and the hidden `--bench-passes` stay in shipped builds.
6. **Accepted-artifact identity.** Engine identity strings changed from
   `cuda precision=bf16x [EXPERIMENTAL, phase 2] ...` to `cuda bf16x kernels=strict(-fmad=false)`.
   Resume therefore re-synthesizes outputs made by the experimental binary once. The audio is
   byte-identical; this is only cache invalidation.
