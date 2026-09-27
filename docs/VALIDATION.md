# Validation and performance

How this implementation was checked against the reference Python pipeline, what its numerical
configuration changes, and what was measured. All measurements are from one host (RTX 4090,
CUDA 12.9, Ubuntu 24.04).

## Reference
The unmodified Python pipeline: kokoro 0.9.4 (`KPipeline`, lang `a`), misaki 0.9.4, spaCy 3.8.14
with en_core_web_sm 3.8.0, eSpeak NG 1.52.0 (as bundled by espeakng-loader 0.2.4), torch 2.12.1,
Kokoro-82M snapshot `f3ff3571`. Exact pins: docs/truth-pack/.

## Text frontend
- Test corpora: frontend edge cases, the full public-domain *Alice* text, link-feature syntax, a
  grammar fuzz corpus, a mixed-script "soup" corpus, and a private long-form text.
- With the reference's eSpeak NG 1.52.0, every line of these corpora gives the reference's phoneme
  strings, chunks and eSpeak NG fallback calls exactly, except the lines refused under the first
  difference below. Where the reference raises an error, the native frontend fails the same line
  with the same error class.
- Two intended differences from the reference:
  - Words the reference leaves without a pronunciation, which it silently drops from the audio,
    make the native frontend fail that line ("unresolved word"). Examples: "-12" in edge case 7,
    the "n’t" of "won’t" on two *Alice* lines, and 135 fuzz lines.
  - The product uses the system eSpeak NG. With a version other than 1.52.0, fallback words can be
    pronounced differently. The exact comparisons above use the reference's 1.52.0 copy.
- Tests: tests/frontend_*.rs, tests/cli_text_native.rs.

## Model
- Correctness was first established with an f32 implementation (since removed). Every
  intermediate stage matched the reference's tensors to ≤ 1e-4 relative L2 on 15 frozen fixture
  cases. End-to-end tolerances were derived from the reference's own run-to-run variation.
- The phoneme-to-id mapping and the voice style vectors used by the current engine still equal the
  reference's exactly on those fixtures (tests/bf16x_reference.rs, which also reports, without
  thresholds, the audio distance of the current engine to the reference).
- The frozen fixtures and the tools that produced them are in `oracle/` and under the test data
  root.

## Numerical configuration
The model runs in one mixed-precision configuration ("BF16x"):
- BF16 tensor-core operands with f32 accumulation for the stride-1 convolutions (a fused
  implicit-GEMM kernel where the shape allows, per-tap cuBLAS GEMMs otherwise) and all linear
  layers;
- f32 for the strided convolution, the 1×1 source convolution, the LSTMs, attention, normalization
  statistics, and the source/STFT/iSTFT stages;
- CUDA kernels compiled with FMA contraction (`-fmad=true`).

Distance from the f32 implementation, measured on 130 public lines as the median per-line spectral
distance:

| configuration | median spectral distance to f32 | durations |
|---|---|---|
| BF16x with strict rounding (`-fmad=false`) | 2.05–2.28 dB | change on some lines |
| other options evaluated and not included: FP16/BF16 decoder only | 0.05 / 0.22–0.25 dB | unchanged |
| FP16 with half-precision predictor | 1.2–1.4 dB | change on some lines |
| TF32 | ~1.1–1.2 dB | change on a few lines |
| INT8 weights and activations | ~3.3 dB | unchanged (audible hiss) |

Quality was judged by listening, comparing with the reference.

Enabling FMA contraction changed every output byte-wise relative to the strict-rounding build. On
the 422 public comparison lines, 98 lines changed length, and the median per-line spectral distance
between the two builds is 1.77 dB (maximum 13.4 dB). tests/strict_reference_diff.rs reports these
differences.

## Performance
Scope:
- One comparison run (all arms in the same session) on the measurement host.
- Built from the source state before FMA contraction was enabled, i.e. the strict-rounding BF16x
  build, and the f32 implementation that has since been removed. The current FMA build has not been
  re-timed.

Benchmark: a private 316-line long-form text (about 49 minutes of audio). The Python baseline is
the unmodified `KPipeline` driven one line at a time by a usage harness (bench/system_reference.py).
It is not a measurement of any particular deployed service.

Median seconds per arm (6 warm passes, 3 cold processes), with the coefficient of variation:

| | Python KPipeline | f32 implementation | BF16x (strict rounding) |
|---|---|---|---|
| warm pass (model resident) | 54.31 s (CV 4.0%) | 7.45 s (1.9%) | 4.15 s (0.6%): 13.1× Python, 1.79× f32 |
| cold process (load + synthesize) | 68.71 s (1.8%) | 8.69 s (2.2%) | 5.81 s (2.9%): 11.8× Python, 1.50× f32 |

- A later, smaller check compared two BF16x builds with byte-identical output (before and after a
  code restructuring). It showed more cold-start variation (CV 5.2%); the two builds' warm passes
  agreed within 1.5%.
- The main sources of speed:
  - length-bucketed batching over a gap-separated ragged layout (docs/design/BATCHING.md);
  - fused BF16 tensor-core convolutions with the AdaIN + Snake prologue fused in;
  - a persistent cooperative LSTM kernel;
  - a parallel frontend stage and pipelined WAV writing;
  - loading the model while the frontend starts.
- The private text itself is not distributed, so these figures cannot be reproduced from this
  repository alone. `kokoro bench` times pure inference on any phoneme chunk file.
