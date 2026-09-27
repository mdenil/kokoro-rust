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

## Attempted and rejected
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
(filled in as measured)
