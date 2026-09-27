# PHASE 2 — lossy reduced-precision / quantization exploration (branch experiment/reduced-precision)

Owner authority: #23 (HERMES_BRIEF.md) + HERMES_PRECISION_EXPLORATION_BRIEF.md.

**Status of every non-f32 level: EXPERIMENTAL and UNREVIEWED.** Only Misha can accept audio quality.
- This branch is not merged and does not change any default on `main`.
- The f32 control is the phase-1 final code (main 3dc5a35). Its preserved binaries are
  `$KOKORO_DATA/bin/phase1-final-{strict,fma}-3dc5a35`.

## Levels (all selected at load via `--precision` / `KOKORO_PRECISION`; recorded in the engine identity)

| level | what runs in reduced precision | operand dtypes | accumulate / output | kept in f32 |
|---|---|---|---|---|
| f32 (control) | nothing | f32 | f32 | everything (phase-1 engine; f32 SIMT fused convs + cuBLAS SGEMM) |
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

## Phase-2 kernel levers (after the matrix; half levels only; each bitwise identical to its predecessor)
Measured with the in-process bench (`kokoro bench`, Alice ch. 1 chunks, af_heart, fp16). Timings are
interleaved rounds of 5 reps, total median, cv ≤ 0.5%. Bitwise identity was checked on Alice ch. 1
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
  Its f32 level is the phase-1 control: chapter cold 8.77 s vs the frozen phase-1 8.79 s.
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
  and 1.32× warm over the phase-1 f32 engine, i.e. 11.3× / 9.1× over production Python. Its
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
