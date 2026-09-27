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
| fp16 | tier 1 = decoder + generator convs (per-tap tensor-core GEMMs) | FP16 activations (transposed [T][Cin] copy per conv) × FP16 weights | f32 accumulate, f32 output | duration/F0/N predictor, text encoder, ALBERT, all linear layers, LSTMs, norms/stats, source/STFT/iSTFT, noise conv |
| bf16 | as fp16 | BF16 × BF16 | f32 | as fp16 |
| fp16x | tier 1 + tier 2 (predictor F0/N resblocks and projections, text-encoder CNN) + ALL linear layers (ALBERT, LSTM input projections, AdaIN style projections, encoders) | FP16 | f32 | LSTM recurrence (h·Whh), norms/stats, attention batched GEMMs, source/STFT/iSTFT, noise conv |
| bf16x | as fp16x | BF16 | f32 | as fp16x |
| int8 | tier 1 convs (decoder + generator) with Cout % 64 = 0 and same-length output (others, e.g. conv_post Cout 22, stay f32) | INT8 weights (symmetric, per-output-channel scale = max\|w\| over (k, Cin) / 127) × INT8 activations (symmetric, dynamic per-tensor scale = max\|x\| / 127 over the whole conv input incl. all batch items) | INT32 accumulate over Cin and taps; f32 dequant + bias | everything else (as fp16) |

Honest notes:
- **Calibration:** none. INT8 weight scales come from the weights; activation scales are dynamic per
  call.
- **Batch dependence:** the per-tensor activation scale spans every item in a batch, so int8 output
  can depend on batch composition.
- **Conversion cost:** half and int8 modes convert and transpose each conv input once per call; that
  cost is inside the measured times.
- **No speed inference from dtype:** only measured wall times count.

## Results
(filled in as measured)
