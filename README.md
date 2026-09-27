# kokoro-rust

Native Rust + CUDA speech synthesis for [hexgrad/Kokoro-82M](https://huggingface.co/hexgrad/Kokoro-82M)
(snapshot `f3ff3571`), American English. One process turns a text file into one WAV per line. The
text frontend (misaki 0.9.4 G2P, spaCy tokenizer/tagger and the espeak-ng fallback) is native;
there is no Python at runtime.

The model runs in the owner-accepted **BF16x** mixed-precision configuration (owner decision #25):
- BF16 tensor-core operands with f32 accumulation for the convolutions and linear layers;
- f32 for LSTM recurrences, attention products, normalization and the source/STFT/iSTFT stages;
- kernels built with strict rounding (`-fmad=false`).
This is the only numerical mode.

Status: main (canonical). Not released, not published, and no license chosen by the owner yet (the
`license` field in Cargo.toml predates any owner decision).

## Requirements
- NVIDIA GPU with compute capability 8.9 (developed on an RTX 4090) and its driver.
- CUDA 12.9 toolkit: `nvcc` at build time; cuBLAS at runtime.
- Model snapshot directory: `config.json`, `kokoro-v1_0.pth`, `voices/`.
- Frontend data directory: `misaki-0.9.4/`, `spacy-en_core_web_sm-3.8.0/`, `espeak-ng-1.52.0/`
  (with `libespeak-ng.so.1.52.0`).
- Optional: `ffmpeg`, only for `--encode`.

Details and licenses of runtime libraries: docs/DEPENDENCIES.md.

## Build
```
cargo build --release          # nvcc from $NVCC or /usr/local/cuda/bin/nvcc
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
  A batch that fails (for example out of memory) is split down to single items automatically;
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
cargo test --release -- --test-threads=2
cargo test --release --test bf16x_golden -- --include-ignored   # + private chapter, local data only
```
- `tests/bf16x_golden.rs` checks, byte for byte, that the build reproduces the owner-accepted
  artifact on pinned public cases.
- `tests/bf16x_reference.rs` pins the model inputs against the Python reference fixtures.
- The frontend and CLI suites cover pronunciation fidelity, line mapping, resume, failure handling
  and the Python-free process audit.
Test data lives under `$KOKORO_DATA` (default `/data/mdenil/code/kokoro-rust`), outside Git.

## History (archival)
- Porting, conformance and performance records: PORT_STATE.md, docs/PERFORMANCE_REPORT.md,
  docs/PERF_LEDGER.md, docs/PHASE2_PRECISION.md, docs/conformance/.
- The BF16x-only cleanup record: docs/RELEASE_CLEANUP.md.
