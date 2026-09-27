# Project history (technical summary)

Historical notes only. They describe how the current product was reached. Nothing here is an
active instruction; the current product and its commands are in README.md.

## 1. Native port, verified against the pinned reference
- Kokoro-82M and its English text pipeline were re-implemented natively in Rust:
  - model: StyleTTS2-style prosody predictor + iSTFTNet decoder;
  - frontend: misaki 0.9.4 G2P, spaCy en_core_web_sm 3.8.0 tokenizer/tagger, espeak-ng 1.52.0
    fallback.
- The reference was the unmodified Python pipeline (kokoro 0.9.4, torch 2.12.1).
- Frontend: phonemes, chunking and fallback calls reproduce the reference exactly on every test
  corpus: frontend edge cases, the full public-domain *Alice* text, link features, a grammar fuzz
  corpus, a "soup" corpus, and a private long-form text. Reference errors are reproduced with the
  same error class.
- Model (original f32 engine, since retired): every intermediate stage matched reference tensors to
  ≤ 1e-4 relative L2 on 15 frozen fixture cases. The end-to-end gates were calibrated against the
  reference's own run-to-run nondeterminism.
- The frozen reference fixtures and the tools that produced them remain in `oracle/` and under the
  test data root.

## 2. Speed work in f32 (retired engine)
- Whole-system comparison on a private 316-line long-form text (about 49 minutes of audio)
  against the Python baseline: the unmodified KPipeline, driven one line at a time by a usage
  harness (bench/system_reference.py).
  - This baseline is a usage harness, not a measurement of any deployed service wrapper.
  - Result: cold process 69.5 → 8.8 s (7.9×), warm resident pass 52.3 → 7.3 s (7.1×), with clean
    run-to-run variation on both sides.
- The largest contributions:
  - length-bucketed batched inference over a gap-separated ragged layout, with per-item
    statistics, style and noise (docs/design/BATCHING.md);
  - fused implicit-GEMM and sliding-window convolution kernels;
  - a persistent cooperative LSTM kernel;
  - a parallel frontend prepare stage with bounded reordering;
  - pipelined WAV writing;
  - overlapping model load with frontend startup.
- Ideas that measured slower or neutral were dropped: larger batches (out of memory), prologue
  fusion in f32, cp.async double buffering, extra host threads, and a 256-channel fused kernel
  (faster, but outside the accuracy bound).

## 3. Reduced-precision study and the selected configuration
Levels tried, each behind a runtime selector at the time:
- TF32 matrix products;
- FP16 / BF16 for the decoder and generator convolutions;
- "x" variants, which also run the predictor, text-encoder convolutions and all linear layers in
  half precision;
- INT8 with per-channel weight scales and dynamic activation scales.

Results on the long-form benchmark (warm pass vs the f32 engine; median spectral distance to f32
on 130 public lines):

| level | warm speed-up vs f32 | spectral distance to f32 | durations |
|---|---|---|---|
| fp16 / bf16 | 1.74× | 0.05 / 0.22–0.25 dB | unchanged |
| **bf16x** | **1.80×** | 2.05–2.28 dB | change on some lines |
| fp16x | 1.76× | 1.2–1.4 dB | change on some lines |
| tf32 | 1.09× | ~1.1–1.2 dB | change on a few lines |
| int8 | 1.37× | ~3.3 dB | unchanged |

- The fused BF16 tensor-core convolution, the AdaIN + Snake prologue fusion and float-partial
  channel statistics account for most of the half-precision speed.
- After a listening review against the reference (whole ladder plus focused worst-case
  comparisons, on raw hash-verified WAVs):
  - **bf16x** was selected and its quality accepted: the listeners found it indistinguishable;
  - **INT8 was rejected** for audible hiss.
- bf16x end to end on the long-form benchmark:
  - Warm: 4.15 s, 1.80× the f32 engine and about 13× the Python KPipeline baseline above.
  - Cold: 5.81 s, about 1.5× / about 12×. Measured in the same final comparison run: 3 cold
    replicates per arm, coefficient of variation 2.9% (bf16x), 2.2% (f32), 1.8% (Python), all
    within the 5% gate.
  - A later, smaller pre/post check of the same bf16x artifact (after the code cleanup) was more
    variable in cold starts: coefficient of variation 5.2% on the chapter. Its warm passes matched
    within 1.5%.

## 4. BF16x-only product
- Everything but the selected configuration was removed: the other precision levels, the f32
  model path, the CPU model backend, the single-item GPU path, every alternative kernel and switch.
- The cleaned build reproduces the accepted bf16x artifact byte for byte: tests/bf16x_golden.rs on
  11 public cases, plus a private long-form case run locally.
- Throughput matched the accepted artifact within 1.5% (warm).
- Configuration was then made explicit and host-independent (docs/PORTABILITY.md).
- Later the CUDA kernels were switched from strict rounding (`-fmad=false`) to compiler FMA
  contraction (`-fmad=true`). Outputs no longer match the strict build byte for byte;
  tests/strict_reference_diff.rs reports the per-line differences (diagnostic).
