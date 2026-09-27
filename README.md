# kokoro-rust

Native Rust + CUDA speech synthesis for [hexgrad/Kokoro-82M](https://huggingface.co/hexgrad/Kokoro-82M)
(snapshot `f3ff3571`), American English. One process turns a text file into one WAV per line. The
text frontend (misaki 0.9.4 G2P, spaCy tokenizer/tagger and the espeak-ng fallback) is native;
there is no Python at runtime.

The model runs in one mixed-precision configuration (BF16x):
- BF16 tensor-core operands with f32 accumulation for the convolutions and linear layers;
- f32 for LSTM recurrences, attention products, normalization and the source/STFT/iSTFT stages;
- CUDA kernels compiled with FMA contraction (`-fmad=true`).
This is the only numerical mode.

## Requirements
Tested scope: Linux x86_64, RTX 4090 (compute capability 8.9), CUDA 12.9, American English. Other
GPUs and platforms are unverified (docs/PORTABILITY.md).
- NVIDIA GPU with compute capability 8.9 (developed on an RTX 4090) and its driver.
- CUDA 12.9 toolkit: `nvcc` at build time; cuBLAS at runtime.
- Model snapshot directory: `config.json`, `kokoro-v1_0.pth`, `voices/`.
- Frontend data directory: `misaki-0.9.4/`, `spacy-en_core_web_sm-3.8.0/`.
- eSpeak NG, installed separately from your distribution (Debian/Ubuntu:
  `sudo apt install libespeak-ng1`, which also installs its data; the `espeak-ng` package works
  too). It is required for text input and pronounces words missing from the lexicon, so those
  pronunciations depend on the installed version. eSpeak NG is GPL-3.0-or-later and is loaded at
  runtime, not included here.
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
Running the test suite needs a separate test data directory; see docs/PORTABILITY.md.

## Documentation
- docs/PORTABILITY.md: configuration, tested scope and test prerequisites.
- docs/DEPENDENCIES.md: runtime dependencies and their licenses.
- docs/design/BATCHING.md: the batched inference layout.
- docs/frontend/: the text frontend specification and coverage.
- docs/truth-pack/: pinned upstream sources and hashes.
- docs/HISTORY.md: conformance checks against the reference, speed work and precision study.
- bench/CORPUS.md: public benchmark and test corpora.
