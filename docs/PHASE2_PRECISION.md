# PHASE 2 — lossy reduced-precision / quantization exploration (branch experiment/reduced-precision)

Owner authority: #23 (HERMES_BRIEF.md) + HERMES_PRECISION_EXPLORATION_BRIEF.md.

## OWNER DECISION (2026-09-27): BF16X SELECTED, presented quality ACCEPTED; INT8 REJECTED
- **bf16x SELECTED; its presented quality ACCEPTED** (owner #25, Misha, 1553678159168151575): "Okay
  BF16x confirmed. These are all indistinguishable to me."
  - Basis: G1-G4 ladder listening plus focused B1-B5 comparisons against production Python.
  - B1-B5 cases, public130: lines 52/af_heart, 1/af_heart, 11/am_adam (top 3 vs_reference spectral
    scores over 266 cases); 42/am_adam and 64/af_heart (worst equal-total-duration per voice).
  - Material: raw, unnormalized paired WAVs from the exact final binary phase2-9b39d48 (sha256
    6fde9d88990a8dc518ec1f366fb17db0869f7e7df5fc1fdea973ea0a371612ea, tree 9b39d48), hash/seal
    verified.
  - Per-case WAV hashes and metrics: `$KOKORO_DATA/evidence/listening/phase2-owner-decision/OWNER_DECISION.json`.
  - bf16x's spectral distances to the Python reference on those five cases are 6.2–15.5 dB. They are
    large mainly because of duration changes, which misalign the comparison. They stay recorded as
    diagnostics; the difference is accepted and NOT to be re-escalated.
- **INT8 REJECTED** (owner #24, 1553673730025197655): hiss/degradation in G3/G4.
- PL-014 phase-1 pairs (G5/G6): no audible differences reported.
- **Other levels** (tf32, tf32all, fp16, bf16, fp16x): heard in the ladder listening, not selected. No
  acceptance is recorded for them.
- **Unchanged:**
  - The f32 baseline and all numerical diagnostics in this report are preserved.
  - Nothing is merged, deployed or made default.
  - Adopting bf16x beyond this branch is a further owner decision.

**Status history:** until the decision above, every non-f32 level was EXPERIMENTAL and UNREVIEWED
(the text below dates from then).
- This branch is not merged and does not change any default on `main`.
- The f32 control runs the phase-1 final code path (main 3dc5a35; byte comparisons of the branch f32
  output are limited to those stated below). Its preserved binaries are
  `$KOKORO_DATA/bin/phase1-final-{strict,fma}-3dc5a35`.

## FINAL RESULTS (binary phase2-9b39d48, tree 9b39d48; ALL non-f32 levels UNREVIEWED)
Binary `$KOKORO_DATA/bin/phase2-9b39d48-6fde9d88990a` (sha256 6fde9d88…12ea): the matrix binary
(c5686da) plus kernel levers P2-L2..L5 (below). Receipt index: `$KOKORO_DATA/evidence/phase2/RECEIPTS.json`
("final").

Validation:
- **f32 control unchanged vs the matrix binary:** the final binary's f32 WAVs are byte-identical to
  the matrix binary's (phase2-c5686da) on the public130 corpus, 130/130 lines, both voices. That is
  the only f32 byte comparison made for the final binary. The f32 level runs the phase-1 code path,
  but it was not byte-compared here against the preserved phase-1 binaries or on other corpora.
- **Test suite:** the recorded invocation (`cargo test --release --features cuda --no-fail-fast --
  --test-threads=2`), full stdout, exit 101. That is identical to main: every target passes except
  the three known enforced failures, whose failure sets match the recorded receipts row by row.
  Private chapter CLI test: exit 0. See `$KOKORO_DATA/evidence/phase2/validation-*/VALIDATION_SUMMARY.md`
  (includes a disclosed contention failure under default test parallelism).
- **Functional:** 0 failures at every level for both voices on public130 and the held-out passages
  (headers, coverage, finiteness); the Python reference also rendered everything.

### Whole system vs the unchanged production Python (fresh matrix, final binary)
- Evidence: `$KOKORO_DATA/evidence/private/phase2/20260927-061505-final-matrix-chapter` (aggregates only) and
  `$KOKORO_DATA/evidence/phase2/20260927-061505-final-matrix-alice`.
- Protocol as before: 3 interleaved cold processes; 2 resident processes × 3 warm passes.
- Coverage ok and no failed runs for every engine. Host load1 3.9–34.8; the > 15 runs were the first
  Python cold (67.08 s, within its cv), the first Rust f32 cold (8.98 s) and one bf16 warm process.

Private chapter (316 lines, ~2915 s of audio; medians):

| engine | cold wall | warm pass | vs Rust f32 (cold / warm) | vs Python (cold / warm) |
|---|---|---|---|---|
| Python (production) | 68.71 s (cv 1.8%) | 54.31 s (cv 4.0%) | — | — |
| Rust f32 control (phase-1 code path) | 8.69 s (cv 2.2%) | 7.45 s (cv 1.9%) | 1.00 / 1.00 | 7.90× / 7.29× |
| tf32 | 8.17 s | 6.82 s | 1.06 / 1.09 | 8.41× / 7.96× |
| **fp16** | **5.79 s** (cv 6.6%) | **4.27 s** (cv 0.8%) | **1.50 (provisional) / 1.74** | **11.86× (provisional) / 12.73×** |
| bf16 | 5.74 s (cv 5.5%) | 4.29 s (cv 2.6%) | 1.51 (provisional) / 1.74 | 11.97× (provisional) / 12.66× |
| fp16x | 5.60 s | 4.24 s | 1.55 / 1.76 | 12.27× / 12.81× |
| bf16x | 5.81 s | 4.15 s | 1.50 / 1.80 | 11.82× / 13.08× |
| int8 | 7.27 s | 5.42 s | 1.20 / 1.37 | 9.45× / 10.03× |

Alice ch. 1 (65 lines, 656 s of audio; Python warm cv 9.2%, so Python ratios there are
PROVISIONAL):

| engine | cold wall | warm pass | vs Rust f32 (cold / warm) | vs Python (cold / warm) |
|---|---|---|---|---|
| Python | 25.69 s (cv 6.5%) | 8.20 s (cv 9.2%) | — | — |
| Rust f32 | 3.35 s | 1.69 s | 1.00 / 1.00 | 7.67× / 4.86× |
| fp16 | 2.90 s (cv 7.3%) | 1.02 s | 1.16 / 1.66 | 8.85× / 8.01× (provisional) |
| bf16 | 2.89 s (cv 12.8%) | 1.02 s | 1.16 / 1.66 | 8.89× / 8.05× (provisional) |
| fp16x | 2.65 s | 1.01 s | 1.26 / 1.67 | 9.69× / 8.12× (provisional) |
| bf16x | 2.61 s | 0.97 s | 1.28 / 1.74 | 9.84× / 8.45× (provisional) |
| tf32 | 3.17 s (cv 10%) | 1.58 s | 1.06 / 1.07 | 8.10× / 5.19× (provisional) |
| int8 | 3.38 s | 1.27 s | 0.99 / 1.33 | 7.59× / 6.46× (provisional) |

- CV gate (≤ 5%) applies to BOTH arms of a ratio. Chapter Python cold 1.8% and warm 4.0% pass.
  - Candidate cold arms fp16 (6.6%) and bf16 (5.5%) do not: their COLD ratios, vs Rust f32 and vs
    Python alike, are PROVISIONAL.
  - Every other chapter arm is ≤ 5% (Rust f32 2.2 / 1.9%; tf32 0.4 / 0.6; fp16x 1.6 / 1.8;
    bf16x 2.9 / 0.6; int8 0.8 / 1.9; cold / warm).
  - Clean headline: chapter WARM fp16 12.73× vs production Python and 1.74× vs Rust f32 (cv: fp16
    0.8%, Python 4.0%, Rust f32 1.9%).
- Alice cold is dominated by model load (~1.4–1.9 s of ~3 s): cold differences there are small and
  noisy.
- int8's host-side weight quantization adds ~0.5 s to model load.

### Quality, final binary (public130; held-out in parentheses; median mean-|ΔdB|)
- Evidence: `…/20260927-061505-final-quality-public130` and `…-final-quality-heldout`.
- The Python reference was re-rendered. Its distance to the (unchanged) Rust f32 moved slightly:
  am_adam 1.490 → 1.503 dB. That is run-to-run variation of the reference itself.

| level | vs f32: af_heart / am_adam | vs Python ref: af_heart / am_adam | lines with changed duration (af / am) |
|---|---|---|---|
| f32 | 0 / 0 | 1.219 / 1.503 (1.283 / 1.632) | — |
| fp16 | 0.047 / 0.049 (0.054 / 0.081); max 0.11 / 0.11 | 1.219 / 1.502 | 0 / 0 |
| bf16 | 0.218 / 0.245 (0.247 / 0.291) | 1.243 / 1.528 | 0 / 0 |
| tf32 (= tf32all) | 1.238 / 1.095 (1.70 / 1.96) | 1.684 / 1.760 | 7 / 5 |
| fp16x | 1.244 / 1.384 (1.86 / 2.15) | 1.659 / 1.903 | 7 / 5 |
| bf16x | 2.051 / 2.275 (2.18 / 2.57) | 2.292 / 2.536 | 45 / 41 (held-out 1 / 1) |
| int8 | 3.324 / 3.363 (3.56 / 3.31) | 3.386 / 3.412 | 0 / 0 |

- vs the matrix binary, fp16/bf16 and every other non-f32 level include P2-L5 (float-partial channel
  statistics). fp16's median distance from f32 went 0.027–0.032 → 0.047–0.049 dB, max 0.049 → 0.11
  dB. Its distance to the Python reference is unchanged (1.219 / 1.502 vs f32's 1.219 / 1.503).
- Scale context only (not a decomposition): the alternate-seed f32 distance is 1.19 / 1.49 dB.

### Recommendation for the owner's listening (UNREVIEWED; no default change)
- **fp16** is the candidate to listen to first:
  - chapter warm 1.74× over the Rust f32 control and 12.73× over production Python (clean cv);
    cold 1.50× / 11.86× is PROVISIONAL (fp16 cold cv 6.6%);
  - 0.05 dB from f32, no timing changes, reference distance unchanged.
- **bf16** is the same speed as fp16 and 5× farther from f32; no reason to prefer it on this GPU.
- **fp16x / bf16x** add ≤ 3.4% speed over fp16 (chapter) but change durations on 5–45 of 130 lines. Poor trade.
- **tf32** gains 6–9% at a ~1.1–1.2 dB distance. Poor trade.
- **int8** is slower than fp16 and farthest from f32. Not recommended without calibration work.
- Listening pack: `$KOKORO_DATA/evidence/listening/phase2-pack-9b39d48` (README, manifests,
  SHA256SUMS). Raw float32 files at every level, both voices, Python reference, worst cases,
  held-out passages, and the PL-014 phase-1 pairs. Status UNREVIEWED.

### Residual opportunities (not pursued; recorded for a later decision)
- WMMA conv still latency-bound: after P2-L2, ncu showed the tensor pipe 30–36% active, warps
  active ~31% at 98 registers/thread, top stall global-load latency. Options: an mma.sync/ldmatrix
  kernel with async (cp.async) f32 staging; tuned tile shapes per layer; fewer registers.
- FP16 accumulation (2× tensor rate on GeForce): a further lossy tier; unmeasured.
- Deterministic fused channel statistics in the conv epilogue (removes the stats read pass,
  ~10% of kernel time at fp16).
- The f32 LSTM recurrence (~10% of kernel time), launch gaps (~10%), GPU-side int8 weight
  quantization (load time).
- Calibrated INT8 (per-channel activation scales from a calibration set).

## Levels (all selected at load via `--precision` / `KOKORO_PRECISION`; recorded in the engine identity)

| level | what runs in reduced precision | operand dtypes | accumulate / output | kept in f32 |
|---|---|---|---|---|
| f32 (control) | nothing | f32 | f32 | everything (phase-1 code path; f32 SIMT fused convs + cuBLAS SGEMM) |
| tf32 | every cuBLAS GEMM: per-tap convs of the 256-ch generator stage, decoder and predictor convs, linear layers | f32 inputs rounded to TF32 (10-bit mantissa) inside tensor cores | f32 | the f32 SIMT fused convs (128-ch generator stage, PL-008/016), LSTM recurrence, norms/statistics, source/STFT/iSTFT, attention batched GEMMs |
| tf32all | as tf32, plus the 128-ch generator convs as per-tap TF32 GEMMs (fused SIMT kernels disabled) | TF32 | f32 | LSTM recurrence, norms/stats, source/STFT/iSTFT, strided noise conv (direct) |
| fp16 | tier 1 = decoder + generator convs: FUSED WMMA tensor-core conv (`conv1d_wmma_f16[_res]`) where Cout % 64 = 0, Cin % 16 = 0 and the output is same-length; other tier-1 convs (e.g. conv_post, Cout 22) as per-tap tensor-core GEMMs | FP16 activations (f32 input converted on the fly into a transposed smem window) × FP16 weights | f32 accumulate, f32 output; Snake-block conv2 accumulates the residual in its epilogue | duration/F0/N predictor, text encoder, ALBERT, all linear layers, LSTMs, norms/stats, source/STFT/iSTFT, noise conv |
| bf16 | as fp16 | BF16 × BF16 | f32 | as fp16 |
| fp16x | tier 1 (fused WMMA as fp16) + tier 2 (predictor F0/N resblocks and projections, text-encoder CNN) + ALL linear layers (ALBERT, LSTM input projections, AdaIN style projections, encoders) | FP16 | f32 | LSTM recurrence (h·Whh), norms/stats, attention batched GEMMs, source/STFT/iSTFT, noise conv |
| bf16x | as fp16x | BF16 | f32 | as fp16x |
| int8 | tier 1 convs (decoder + generator) with Cout % 64 = 0, Cin % 16 = 0 and same-length output, via the FUSED WMMA int8 kernel (`conv1d_wmma_s8[_res]`); other convs (e.g. conv_post, Cout 22) stay f32 | INT8 weights (symmetric, per-output-channel scale = max\|w\| over (k, Cin) / 127) × INT8 activations (symmetric, dynamic per-tensor scale = max\|x\| / 127 over the whole conv input incl. all batch items) | INT32 accumulate over Cin and taps; f32 dequant + bias | everything else (as fp16) |

Honest notes:
- **Calibration:** none. INT8 weight scales come from the weights; activation scales are dynamic per
  call.
- **Batch dependence:** the per-tensor activation scale spans every item in a batch, so int8 output
  can depend on batch composition.
- **Conversion cost:** the fused kernels convert f32 inputs while staging each tile; the per-tap fallback
  converts and transposes each conv input once per call. Both costs are inside the measured times.
- **No speed inference from dtype:** only measured wall times count.

## Phase-2 kernel levers (after the matrix; non-f32 levels only; P2-L2..L4 bitwise identical to their predecessors on the checked set below, P2-L5 not bitwise)
Measured with the in-process bench (`kokoro bench`, Alice ch. 1 chunks, af_heart, fp16). Timings are
interleaved rounds of 5 reps, total median. Per-process cv ≤ 0.65% for every kept lever (max: P2-L5
receipt sf-1-1 0.646%); the neutral staging-depth A/B reached 0.79%. Bitwise identity was checked on Alice ch. 1
× {af_heart, am_adam} × {fp16, bf16}: 65/65 WAVs byte-identical each.

| lever | change | fp16 forward | verdict |
|---|---|---|---|
| (matrix binary) | v1 fused WMMA conv | 1.105 s | baseline |
| P2-L2 | v2 WMMA conv: A fragments straight from a pre-laid-out global weight buffer (no weight smem staging); 32 input channels per __syncthreads pair; 2×2 warp layout (32 co × 64 t per warp: 2 A + 4 B loads per 8 MMAs); per-warp 16×16 epilogue tile instead of 32 KB block staging. Kill switch KOKORO_LP_WMMA2=0 | 0.933 s (1.18×) | KEEP |
| P2-L3 | v2 with 128-co blocks (8 warps): each staged input window feeds twice the MMAs; the 128-ch stage stages its input once. Kill switch KOKORO_LP_WMMA2W=0 | 0.915 → 0.895 s (−2.2%) | KEEP |
| P2-L4 | AdaIN + Snake applied inside the v2 conv's input staging (exactly adain_apply_seg's float expression; `-fmad=false` build) instead of a separate apply kernel: saves one activation write + read per AdaIN in every Snake block. Kill switch KOKORO_LP_PROLOGUE=0 | 0.895 → 0.855 s (−4.5%) | KEEP |
| (neutral) | v2 staging with each warp streaming whole channel rows (no per-element divides, unrolled loads) | 0.933 vs 0.931 s | NEUTRAL, reverted |
| P2-L5 | channel statistics with float per-thread partials over shifted values (x − segment's first element), combined in double. The PL-014 kernel is FP64-bound on GeForce (ncu: SM 71.5%, DRAM 40%). Non-f32 levels only (the f32 control keeps PL-014). Kill switch KOKORO_LP_STATS_F32=0 | 0.856 → 0.823 s (−3.9%) | KEEP, **approximately lossless, NOT bitwise**: fp16 vs f32 on Alice median 0.030 → 0.048 dB, max 0.039 → 0.127 dB; vs the pre-L5 fp16 median 0.045 dB, rel 4.4e-3. This is the size of the PL-014 reordering drift (3.9e-3), i.e. rounding-level changes amplified through the F0/phase path |
| (neutral) | v2 with 64 input channels (4 slices) per stage (XLD 80) | 0.822 vs 0.822 s | NEUTRAL, reverted |

- Cumulative fp16 forward (in-process core timing; NOT a whole-system or production ratio):
  1.105 → 0.855 s through P2-L4, output bitwise identical to the matrix binary; 0.823 s with P2-L5,
  which changes output (approximately lossless).
- Whole-system numbers and quality for the final binary come only from the fresh matrix and quality
  sweep in the final validation below. Core gains are not multiplied into production ratios.
- Hardware counters after P2-L2 (ncu, same method as above): tensor pipe 30–36% active (v1 ~20%),
  DRAM 48–57%, warps active ~31% (98 registers/thread), top stall long-scoreboard (global loads).

## Attempted and rejected
- **Half WMMA for any Cin/Cout** (KOKORO_LP_WMMA_ANY=1: route the decoder convs, Cin 514 / 1090,
  and conv_post through the fused kernel instead of the per-tap fallback; the kernel already
  zero-pads partial tiles). NEUTRAL: fp16 Alice forward 1.106 → 1.100 s (−0.5%, 3 interleaved
  rounds of 5 reps, cv ≤ 0.4%). Kept opt-in, default off.
- **Per-tap cuBLAS tensor-core GEMMs** for fp16/bf16/tf32all: correct but SLOWER than the phase-1 f32
  fused kernels. Alice forward 2.0–2.1 s vs f32 1.49 s. Per-tap GEMMs re-read/write the output for every
  tap: memory-bound, the problem PL-008 solved. Replaced by the fused WMMA kernel; per-tap stays only as
  a fallback for shapes the fused kernel does not take.
- **cuBLAS per-tap INT8 (IMMA via cublasGemmEx)**: CUBLAS_STATUS_NOT_SUPPORTED for our shapes; replaced
  by the fused WMMA int8 kernel.
- **INT8 with per-time-step ("per-token") activation scales (int8t)**: implemented, measured 8.6 dB
  spectral distance vs f32 (worse than per-tensor 4.3 dB). This is mathematically invalid for
  convolutions: an output sums over taps at different input time steps, so per-input-column scales
  cannot be factored out after the int32 accumulation. Reverted; not in the ladder.
- **Single-block absmax** for the int8 dynamic scale: correct but 4.15 s forward (serial reduction over
  the whole conv input); replaced by a multi-block atomic-max reduction (1.19 s).

## Results

### Quality diagnostics (public130 = Alice ch. 1, 65 lines + frontend edge cases, 65 lines)
- Run: `$KOKORO_DATA/evidence/phase2/20260927-042244-quality-public130` (manifest.json holds every
  per-line metric and WAV hash).
  - Binary `phase2-c5686da` sha256 10739ae5…a496, tree c5686da; corpus sha256 cd7566a5…31a3.
  - Voices af_heart and am_adam, speed 1.0, excitation seed 0.
- Metric: mean |ΔdB| of the log spectrum (Hann 1024 / hop 256). Waveform rel is misleading here: an
  F0 or phase shift gives rel ≈ 1 while the audio can sound identical.
- **Scale for these numbers:** the Rust f32 control vs the Python reference is 1.22 / 1.49 dB. The
  reference draws its excitation noise from torch's RNG, so it never shares Rust's noise. For scale
  only, a same-engine alternate-seed distance (f32 seed 1 vs seed 0) is given below. It does NOT show
  how much of any cross-engine difference is caused by noise.
- Functional failures: **0** at every level for both voices: every line rendered, 24 kHz mono float
  WAV, finite, non-empty. The Python reference also rendered all 130 lines per voice without failures.

| level | af_heart: median / max dB vs f32 | af_heart: median dB vs Python ref | am_adam: median / max dB vs f32 | am_adam: median dB vs Python ref | lines with changed duration (af / am) |
|---|---|---|---|---|---|
| f32 (control) | 0 / 0 | 1.218 | 0 / 0 | 1.490 | — |
| fp16 | 0.027 / 0.039 | 1.219 | 0.032 / 0.049 | 1.490 | 0 / 0 |
| bf16 | 0.211 / 0.274 | 1.242 | 0.238 / 0.463 | 1.516 | 0 / 0 |
| tf32 | 1.227 / 9.61 | 1.674 | 1.096 / 7.90 | 1.754 | 7 / 5 |
| tf32all | ≈ tf32 (1.227 / 9.61) | 1.674 | ≈ tf32 | 1.754 | 7 / 5 |
| fp16x | 1.247 / 9.61 | 1.652 | 1.384 / 9.74 | 1.888 | 7 / 5 |
| bf16x | 2.049 / 15.6 | 2.294 | 2.285 / 11.3 | 2.553 | 45 / 41 |
| int8 | 3.321 / 4.86 | 3.382 | 3.355 / 4.48 | 3.418 | 0 / 0 |

**Seed-floor calibration** (`$KOKORO_DATA/evidence/phase2/20260927-051206-seedfloor-public130`):
- The same f32 engine rendered with excitation seed 1 vs seed 0 is 1.188 / 1.486 dB apart (median;
  max 1.35 / 1.69), af_heart / am_adam.
- This is noise-scale context only. The f32-vs-Python distance (1.22 / 1.49 dB) is of the same size,
  but that does not show that noise explains most of the cross-engine difference: the metric is not
  additive, and other differences can hide inside a distance of that size.
- Relative to that scale (a comparison of magnitudes, not a decomposition):
  - fp16's distance from f32 is about 1/40 of it;
  - bf16's about 1/5;
  - tf32 / fp16x about 1× (plus timing changes on a few lines);
  - int8's about 2.5×, with durations and noise draws identical to f32.
- The f32 seed-0 renders reproduced the earlier quality run bitwise (130/130 lines, both voices).

**Held-out passages** (`bench/corpus_heldout.txt`, sha256 0e7d9a68…3326: Alice ch. 2, Pride and
Prejudice, one synthetic line; used in no development corpus):
- Run: `$KOKORO_DATA/evidence/phase2/20260927-051206-quality-heldout`, Python reference included.
- Median dB vs f32, af_heart / am_adam: fp16 0.031 / 0.032; bf16 0.237 / 0.277; tf32 1.70 / 1.96;
  fp16x 1.86 / 2.15; bf16x 2.19 / 2.57 (one line with a changed duration per voice); int8 3.57 / 3.31.
- f32 vs Python 1.27 / 1.63 dB. 0 functional failures. The ordering matches public130.

Reading (diagnostic, not acceptance):
- **fp16** is numerically the closest level by far: 0.03 dB from f32, and its distance to the Python
  reference is unchanged.
- **bf16** costs about 0.2 dB (7-bit mantissa).
- **tf32 / fp16x / bf16x** round the predictor path (ALBERT / linears / predictor GEMMs). Their
  durations change on some lines and F0 shifts. Max values on duration-changed lines compare
  misaligned audio, so they mean "different timing", not necessarily audible loss.
- **int8** keeps the durations exactly (only the decoder/generator is quantized) but adds a broadband
  ~3.3 dB spectral distance. It is the most likely to be audibly degraded; listen before drawing any
  conclusion.

### Speed: whole system, cold process and warm resident, vs the unchanged production Python
Setup:
- Harness `bench/system_compare.py`: same protocol as phase 1, interleaved engines, 3 cold reps, 2 × 3
  warm passes, idle-GPU wait, host load recorded.
- One binary, `phase2-c5686da` (sha256 10739ae5…a496), with the level selected by KOKORO_PRECISION.
  Its f32 level runs the phase-1 code path (timing check only: chapter cold 8.77 s vs the frozen
  phase-1 8.79 s; not a byte comparison).
- Evidence: private chapter `$KOKORO_DATA/evidence/private/phase2/20260927-043001-matrix-chapter`
  (identity + raw only; contents never published); Alice
  `$KOKORO_DATA/evidence/phase2/20260927-043001-matrix-alice`; SHA256SUMS in each; index
  `$KOKORO_DATA/evidence/phase2/RECEIPTS.json`.
- Coverage ok and no failed runs for every engine.

Private chapter (316 lines, ~2915 s of audio; medians; Rust cv ≤ 2.5% except where shown):

| engine | cold wall | warm pass | speedup vs Rust f32 (cold / warm) | vs Python (cold / warm) |
|---|---|---|---|---|
| Python (production, unchanged) | 78.08 s (cv 11.6%, host load 25–28 during the first reps) | 50.38 s (cv 6.9%) | — | — |
| Rust f32 (control) | 8.77 s | 7.32 s | 1.00 / 1.00 | 8.90× / 6.89× |
| tf32 | 8.55 s | 6.98 s | 1.03 / 1.05 | 9.13× / 7.22× |
| **fp16** | **6.92 s** | **5.55 s** | **1.27 / 1.32** | **11.28× / 9.08×** |
| bf16 | 7.13 s | 5.64 s | 1.23 / 1.30 | 10.95× / 8.93× |
| fp16x | 6.80 s | 5.48 s | 1.29 / 1.34 | 11.48× / 9.19× |
| bf16x | 6.95 s | 5.58 s (cv 4.8%) | 1.26 / 1.31 | 11.24× / 9.03× |
| int8 | 7.37 s | 5.54 s | 1.19 / 1.32 | 10.59× / 9.09× |

Alice ch. 1 (65 lines, 656 s of audio):

| engine | cold wall | warm pass | vs Rust f32 (cold / warm) |
|---|---|---|---|
| Python | 25.87 s (cv 12.5%) | 8.11 s (cv 5.6%) | — |
| Rust f32 | 3.26 s (cv 5.4%) | 1.69 s | 1.00 / 1.00 |
| tf32 | 3.18 s | 1.61 s | 1.03 / 1.05 |
| fp16 | 3.20 s (cv 8.6%) | 1.32 s | 1.02 / 1.28 |
| bf16 | 2.90 s (cv 11.5%) | 1.33 s (cv 5.1%) | 1.12 / 1.27 |
| fp16x | 3.13 s (cv 7.0%) | 1.28 s | 1.04 / 1.32 |
| bf16x | 2.85 s | 1.29 s | 1.14 / 1.31 |
| int8 | 3.34 s | 1.30 s | 0.98 / 1.30 |

Notes:
- Python-relative ratios carry the Python arm's cv (11–12% cold, host contention recorded in
  raw.jsonl). Ratios against Rust f32 are the clean comparison.
- Alice cold is dominated by model load and CUDA init (~1.3–1.9 s of ~3 s), so the cold differences
  there are within noise. Warm Alice and both chapter rows are clean.
- **int8 load cost:** model_load 1.86–1.88 s vs 1.33–1.48 s. INT8 weight quantization runs on the
  host at load. It could move to the GPU; not done, since int8 has no speed advantage over fp16 to
  justify it.
- **Memory:** the half levels keep the f32 weights too, adding half copies of the tier-1 conv weights
  (+~2 bytes per parameter of the decoder/generator convs, on the order of +100 MB of device memory
  at most). Not separately measured.

### Summary for the owner (matrix binary phase2-c5686da; ALL UNREVIEWED; Python ratios PROVISIONAL: Python arm cv 11–12% under recorded host contention)
- **fp16** is the clear candidate. It captures nearly all of the available speed: chapter 1.27× cold
  and 1.32× warm over the Rust f32 control, i.e. 11.3× / 9.1× over production Python. Its
  spectral distance from f32 was 0.03 dB in the matrix binary (1/40 of the alternate-seed distance,
  scale context only), with no timing change on any of 133 lines × 2 voices. It remains UNREVIEWED
  until listened to. Later kernel levers (below) change these numbers: see the final validation.
- **bf16** is slightly slower than fp16 and 8× farther from f32. No reason to prefer it on this GPU.
- **fp16x / bf16x** gain only 1–2% more than fp16 (the linears/predictor are a small share of time).
  They change the predicted durations and F0 on some lines. Poor trade.
- **tf32** is 3–5% faster than f32 but its distance from f32 is about the size of the
  alternate-seed distance (predictor GEMMs).
  Poor trade; tf32all is identical in quality and slower.
- **int8** (no calibration, dynamic per-tensor activation scales) is no faster than fp16 warm, slower
  cold, and the farthest from f32 (~3.3 dB, 2.5× the alternate-seed distance, broadband). Not recommended
  without calibration work.
- **Where time goes now (measured):**
  - Whole-system warm pass at fp16 (Alice, `--timeline`): the GPU stage is 1.09 of 1.28 s, still
    85% of the pass (f32: 1.48 of 1.68 s).
  - Kernel breakdown (nsys, in-process bench, Alice ×4 forwards):
    - total 5.60 s → 4.05 s f32 → fp16;
    - at fp16: fused WMMA convs 48% (res 27%, plain 21%), AdaIN apply 11%, channel stats 10%,
      LSTM 8%, everything else ≤ 4% each.
  - Hardware counters (ncu, first 40 WMMA launches of an 8-chunk fp16 run, time-weighted):
    - tensor pipe (HMMA) active 22.5% (conv1d_wmma_f16) / 18.6% (_res) of peak sustained;
    - SM throughput 32% / 26%; DRAM 27% / 32%.
    The kernels are latency-bound: single-buffered 16-channel staging with two __syncthreads per
    slice and 4 warps/block. Neither compute- nor bandwidth-bound, so there is real headroom.
  - Remaining levers: (1) fuse AdaIN apply + Snake into the WMMA input staging (removes one
    activation write + read per conv); (2) a more efficient WMMA pipeline (double-buffered staging,
    larger tiles); (3) FP16 accumulation (2× tensor rate on GeForce; an aggressive-tier lossy level).
- **Memory:** the half levels keep the f32 weights and add half copies of the tier-1 conv weights.
  Upper bound: +2 bytes × 82M parameters ≈ +164 MB device memory. Peak host RSS unchanged (931 vs
  928 MB). Device memory not separately measured.

Listening pack: `$KOKORO_DATA/evidence/listening/phase2-pack` (README.md, SHA256SUMS; raw float32).
- public130 cases, the worst per level, held-out passages, and the PL-014 phase-1 pairs.
- 501 files, 516 MB, status UNREVIEWED.

### Documentation corrections (supervisor final audit; docs only)
- The sealed listening-pack READMEs (not edited, to keep their seals) say two things too broadly:
  - "f32 is bitwise identical in the final binary": verified only vs the matrix binary on
    public130 (130/130, both voices);
  - the PL-014 export's "the f32 path is the main code": that export was built from the branch tree.
    Its metrics match the ledgered PL-014 rows to the printed 4 decimals; it was not byte-compared
    against a main-built export.
- Cold fp16/bf16 chapter ratios are provisional (candidate cold cv 6.6% / 5.5%). See Final results.
