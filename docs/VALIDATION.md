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
The current build (BF16x with FMA contraction, one WAV per input) against the unmodified Python
pipeline, as whole commands on the public text in this repository.

- Input: `bench/corpus_alice_ch1-3.txt`, chapters I–III of *Alice's Adventures in Wonderland*
  (251 lines; sha256 `8108bc77…`; [bench/CORPUS.md](../bench/CORPUS.md)), voice `af_heart`.
- Both write the whole input as one 24 kHz 16-bit WAV: kokoro-rust with its default command,
  Python with `KPipeline(lang_code='a')` (kokoro 0.9.4, torch 2.12.1, on the same GPU) called line by
  line, the audio of all lines joined and written with `soundfile` at the end
  (`bench/system_reference.py --single-wav`).
- Each timed run is a fresh process with a fresh output directory, so nothing is reused. The time
  covers everything: starting the process (and importing Python and torch), loading the model and
  the text frontend, synthesis, and writing the WAV.
- Disk cache warm: each program ran once untimed first, so its files were in the page cache. The
  timed runs then alternated between the two programs.
- Host: RTX 4090, CUDA 12.9, driver 580, Ubuntu 24.04, 80 CPU threads. No other GPU process was
  seen in the snapshots taken at the start and end of each run (the GPU was not monitored in
  between). Other CPU work on the shared host: 1-minute load average 4.4–11.4.
- CPU settings: Python used torch's default of 40 CPU threads; kokoro-rust its defaults (16 text
  preparation threads, 4 writer threads).
- Build: kokoro-rust commit `d94db53` (binary sha256 `8ae5879e…`, the same binary as the tested
  commit `a389b1a`).

| whole command, disk cache warm (5 runs each) | Python KPipeline | kokoro-rust |
|---|---|---|
| median | 52.46 s | 5.06 s |
| range | 49.98–55.62 s | 4.95–5.94 s |
| all runs | 52.46, 51.57, 55.62, 53.77, 49.98 | 4.95, 5.94, 5.51, 5.06, 5.01 |

The ratio of the medians is 10.4; the coefficient of variation of the runs is 4.1% (Python) and
8.0% (kokoro-rust). The outputs were 44,808,600 samples (Python) and 44,799,000 samples
(kokoro-rust), both 31.1 minutes. The two differ in numerics (reduced precision, see above) and in
the eSpeak NG used for words missing from the dictionary: the installed 1.51 for kokoro-rust, and
the 1.52.0 bundled with the Python package.

To reproduce (with the Python reference environment of `scripts/setup_reference_env.sh` and the
test data root of [PORTABILITY.md](PORTABILITY.md)):
```
KOKORO_DATA=/path/to/data python3 bench/system_compare.py --corpus bench/corpus_alice_ch1-3.txt \
    --out results --output single --engines py,rust --phases cold --cold-reps 5 --cache-warmup
```

The main sources of speed:
- length-bucketed batching over a gap-separated ragged layout ([design/BATCHING.md](design/BATCHING.md));
- fused BF16 tensor-core convolutions with the AdaIN + Snake prologue fused in;
- a persistent cooperative LSTM kernel;
- a parallel frontend stage and pipelined WAV writing;
- loading the model while the frontend starts.

### Earlier measurements
- The same build on chapter I only (`bench/corpus_alice_ch1.txt`, 65 lines, 10.9 minutes of audio),
  5 fresh processes each without a warm-up run: medians 25.19 s (Python, 23.63–30.69) and 2.67 s
  (kokoro-rust, 2.57–2.89). With the model already loaded, in benchmark harnesses that time
  repeated passes: 8.01 s and 1.33 s (6 passes each).
- Before FMA contraction was enabled, the strict-rounding BF16x build was compared with the Python
  pipeline and with the since-removed f32 implementation on a private 316-line long-form text
  (about 49 minutes of audio, one WAV per line). Medians: whole process 68.71 s (Python), 8.69 s
  (f32), 5.81 s (BF16x); model loaded 54.31 s, 7.45 s and 4.15 s. That text is not distributed, so
  these figures can't be reproduced from this repository.
