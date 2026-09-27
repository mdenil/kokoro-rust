# kokoro-rust

Native Rust + CUDA speech synthesis for [hexgrad/Kokoro-82M](https://huggingface.co/hexgrad/Kokoro-82M)
(snapshot `f3ff3571`), American English. One process turns a text file into one WAV per line. The
text frontend (misaki 0.9.4 G2P, spaCy tokenizer/tagger and the espeak-ng fallback) is native;
there is no Python at runtime.

The model runs in the **BF16x** mixed-precision configuration, selected after a listening review
(docs/HISTORY.md):
- BF16 tensor-core operands with f32 accumulation for the convolutions and linear layers;
- f32 for LSTM recurrences, attention products, normalization and the source/STFT/iSTFT stages;
- CUDA kernels compiled with FMA contraction (`-fmad=true`).
This is the only numerical mode.

Status: not released or published. No project license has been decided yet: the `license` field in
Cargo.toml predates that decision.

## Requirements
Tested scope: Linux x86_64, RTX 4090 (compute capability 8.9), CUDA 12.9, American English. Other
GPUs and platforms are unverified (docs/PORTABILITY.md).
- NVIDIA GPU with compute capability 8.9 (developed on an RTX 4090) and its driver.
- CUDA 12.9 toolkit: `nvcc` at build time; cuBLAS at runtime.
- Model snapshot directory: `config.json`, `kokoro-v1_0.pth`, `voices/`.
- Frontend data directory: `misaki-0.9.4/`, `spacy-en_core_web_sm-3.8.0/`, `espeak-ng-1.52.0/`
  (with `libespeak-ng.so.1.52.0`).
- Optional: `ffmpeg`, only for `--encode`.

Details and licenses of runtime libraries: docs/DEPENDENCIES.md.

## Build
```
cargo build --release          # nvcc: $NVCC, $CUDA_HOME/bin, $CUDA_PATH/bin, PATH, /usr/local/cuda/bin
```

## Use
```
kokoro synth --model-dir <snapshot> --frontend-dir <frontend-data> \
    --input book.txt --out-dir out/ [--voice af_heart] [--speed 1.0]
```
- Output: `out/book_<line>.wav` (24 kHz mono, pcm16 or `--format float32`) plus a JSON sidecar per
  line, and `out/book.manifest.json`.
- Line numbers are 1-based and never shift. Failures are recorded per line (exit 1 if any line
  failed).
- Re-running resumes: outputs are reused only when their recorded audio hash verifies and nothing
  that affects them changed. `--force` re-synthesizes everything.
- `--model-dir` / `--frontend-dir` can come from `KOKORO_MODEL_DIR` / `KOKORO_FRONTEND_DIR`.

Other options, all operational (they do not change the numerical mode):
- `--cuda-device`;
- batching budget: `--batch-phonemes`, `--batch-items`, `--batch-window`, `--batch-first-window`.
  A batch that fails (for example out of memory) is split down to single items automatically.
  Because batch shape affects the BF16 predictor, such a split can change the audio of those
  lines, so outputs are byte-reproducible only when no split occurs (see docs/design/BATCHING.md);
- `--input-format phonemes`;
- `--seed`;
- `--blank-lines`;
- worker threads: `--prep-threads`, `--write-threads`;
- `--fsync`;
- `--encode <ext>` (external ffmpeg);
- `--timeline <json>`.

`kokoro bench --model-dir <snapshot> --chunks <phonemes.jsonl>` times pure inference.

## Tests
```
KOKORO_DATA=/path/to/data cargo test --release -- --test-threads=2
KOKORO_DATA=/path/to/data cargo test --release --test strict_reference_diff -- --ignored --nocapture   # diagnostic
```
- `tests/bf16x_regression.rs`: run-to-run reproducibility and negative controls.
- `tests/strict_reference_diff.rs` (diagnostic): per-line differences from the earlier
  strict-rounding (`-fmad=false`) build.
- `tests/bf16x_reference.rs` pins the model inputs against the Python reference fixtures.
- The frontend and CLI suites cover pronunciation fidelity, line mapping, resume, failure handling
  and the Python-free process audit.
Tests read their data root from `KOKORO_DATA` (required; outside Git). The GPU is the caller's
`CUDA_VISIBLE_DEVICES` selection. They also need `strace` and `ffmpeg` on PATH and a non-pinned
libespeak-ng for one negative control. Details and optional overrides: docs/PORTABILITY.md.

## Documentation
- docs/PORTABILITY.md: configuration, tested scope and test prerequisites.
- docs/DEPENDENCIES.md: runtime dependencies and their licenses.
- docs/design/BATCHING.md: the batched inference layout.
- docs/frontend/: the text frontend specification and coverage.
- docs/truth-pack/: pinned upstream sources and hashes.
- docs/HISTORY.md: how the product was reached (conformance, speed work, precision study).
- bench/CORPUS.md: public benchmark and test corpora.
