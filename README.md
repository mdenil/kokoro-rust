# kokoro-rust

A native Rust + CUDA implementation of the [Kokoro-82M](https://huggingface.co/hexgrad/Kokoro-82M)
text-to-speech model for American English. It turns a UTF-8 text file with one utterance per line
into one 24 kHz mono WAV file per line, in a single process on an NVIDIA GPU. The text frontend
(misaki G2P, a spaCy tokenizer/tagger, and eSpeak NG for words missing from the lexicon) is
implemented natively as well, so no Python is needed to synthesize speech.

It is a port of the reference Python pipeline ([hexgrad/kokoro](https://github.com/hexgrad/kokoro)
0.9.4 with misaki 0.9.4) and was checked against it; see [docs/VALIDATION.md](docs/VALIDATION.md).

## Requirements
Tested on Ubuntu 24.04 (x86_64) with an RTX 4090, NVIDIA driver 580, CUDA 12.9 and Rust 1.98.
Other GPUs, distributions and versions are untested ([docs/PORTABILITY.md](docs/PORTABILITY.md)).

- An NVIDIA GPU with compute capability 8.9 (Ada, e.g. RTX 40-series) and its driver. Older GPUs
  are refused at startup; newer ones may work but are unverified.
- The CUDA toolkit: `nvcc` to build, cuBLAS at run time.
- A Rust toolchain (`cargo`).
- eSpeak NG, installed from your distribution. On Debian/Ubuntu:
  `sudo apt install libespeak-ng1` (this also installs its data; the `espeak-ng` package works too).
- For the one-time asset download: `bash`, `curl`, `unzip`, `sha256sum`, and Python 3.12 with the
  `venv` module (used once to export spaCy data; not needed afterwards).
- Optional: `ffmpeg`, only for `--encode`.

## Installation
Build:
```
git clone https://github.com/mdenil/kokoro-rust.git
cd kokoro-rust
cargo build --release
```
The binary is `target/release/kokoro`. The build looks for `nvcc` in `$NVCC`, `$CUDA_HOME/bin`,
`$CUDA_PATH/bin`, `PATH`, then `/usr/local/cuda/bin`.

Download the model and language data into a directory of your choice (about 500 MB, of which the
166 MB Python environment under `<dir>/tools/` can be deleted once setup has succeeded):
```
scripts/fetch_assets.sh ~/kokoro-data
scripts/prepare_spacy_assets.sh ~/kokoro-data
```
- `fetch_assets.sh` downloads the Kokoro-82M weights, config and the voices `af_heart` and
  `am_adam` from Hugging Face (revision `f3ff3571`), and the misaki 0.9.4 lexicons from PyPI.
- `prepare_spacy_assets.sh` creates a Python virtual environment with exact package versions from PyPI
  and the `en_core_web_sm` 3.8.0 model from GitHub, and exports the tokenizer and tagger data.
- Checked against pinned SHA-256 hashes: the model files, the misaki and `en_core_web_sm` wheels,
  the extracted lexicons and the six exported spaCy files. The Python packages of the export
  environment are pinned to exact versions only, not hashes; the exported files are what is
  verified. Both scripts can be re-run; files that already verify are kept.
- eSpeak NG is not downloaded; it comes from your system (see Requirements).

## Usage
```
target/release/kokoro synth \
    --model-dir ~/kokoro-data/model --frontend-dir ~/kokoro-data/frontend \
    --input book.txt --out-dir out/
```
`--model-dir` and `--frontend-dir` can also be given as `KOKORO_MODEL_DIR` and
`KOKORO_FRONTEND_DIR`.

Input: UTF-8 text, one utterance per line. Blank lines are errors by default
(`--blank-lines skip` records them without audio instead).

Output, for `book.txt`:
- `out/book_00001.wav`, `out/book_00002.wav`, …: one file per input line, numbered from 1, 24 kHz
  mono, 16-bit PCM (`--format float32` for 32-bit float).
- `out/book_00001.json`, …: a sidecar per line with the text, phonemes, status, any error, the
  audio hash and the configuration that produced it.
- `out/book.manifest.json`: a summary of the run.

Exit status: 0 when every line was synthesized, 1 when some lines failed (each failure is in its
sidecar and in the manifest), 2 for errors that stop the whole run (missing files, no usable GPU,
missing eSpeak NG).

Running the same command again resumes: a line is skipped only if its WAV still matches the
recorded hash and nothing that affects it (text, voice, speed, model, frontend data, eSpeak NG
installation, …) has changed. `--force` re-synthesizes everything.

Common options:
- `--voice am_adam`: voice (default `af_heart`); a `.pt` path or `a,b` (average) also work. The
  setup script downloads only `af_heart` and `am_adam`, the two tested voices.
- `--speed 1.1`: speaking rate.
- `--encode flac` (or `mp3`, `opus`, …): also encode each WAV with `ffmpeg`.
- `--cuda-device N`: GPU index among the visible devices.
- `--input-format phonemes`: input lines are already phoneme strings (no frontend, no eSpeak NG).
- `--seed N`: seed of the model's noise excitation (default 0).
- Batching and threading (`--batch-phonemes`, `--batch-items`, `--batch-window`,
  `--prep-threads`, `--write-threads`): performance tuning only.

`kokoro synth --help` lists everything.

## Limitations
- American English only, GPU only (there is no CPU mode), one tested GPU model.
- A line that contains a word the frontend cannot pronounce is not synthesized; the line is
  reported as failed ("unresolved word") instead of being read with the word left out. Examples:
  negative numbers written as "-12", words in non-Latin scripts, some contractions with typographic
  apostrophes after certain words. Rewrite such words (e.g. "minus 12") and re-run.
- Words missing from the lexicon are pronounced by eSpeak NG, so their pronunciation can differ
  between eSpeak NG versions.
- A line whose phonemes cannot be split into chunks of at most 510 characters is refused, not
  truncated.
- Audio depends slightly on which lines are synthesized together in a batch. For the same input and
  options, output is byte-identical from run to run, unless GPU memory runs short and a batch has to
  be split, which prints a message. See [docs/design/BATCHING.md](docs/design/BATCHING.md).
- Numerical mode: BF16 tensor-core arithmetic (f32 accumulation) for convolutions and linear layers,
  f32 elsewhere. Output is close to, but not identical with, the reference Python pipeline
  ([docs/VALIDATION.md](docs/VALIDATION.md)).

## License
The license of this project's own code has not been decided yet. (Cargo.toml contains
`license = "Apache-2.0"`; that field is not a decision and may change.)

Third-party components keep their own licenses (details in [docs/THIRD_PARTY.md](docs/THIRD_PARTY.md)):
- Kokoro-82M weights and voices: Apache-2.0.
- misaki lexicons: Apache-2.0.
- spaCy and en_core_web_sm: MIT.
- eSpeak NG: GPL-3.0-or-later. It is not included here; the program loads the installed system
  library at run time.
- CUDA driver and cuBLAS: NVIDIA's license terms. They are not included here.

## Documentation
- [docs/PORTABILITY.md](docs/PORTABILITY.md): configuration reference, tested scope, and how to run the tests.
- [docs/DEPENDENCIES.md](docs/DEPENDENCIES.md): what the program loads at run time.
- [docs/THIRD_PARTY.md](docs/THIRD_PARTY.md): third-party components, origins and licenses.
- [docs/VALIDATION.md](docs/VALIDATION.md): how the port was checked against the reference, and measured performance.
- [docs/design/BATCHING.md](docs/design/BATCHING.md): batched inference.
- [docs/frontend/](docs/frontend/): the text frontend specification and coverage.
- [docs/truth-pack/](docs/truth-pack/): the pinned upstream sources and hashes.
- [bench/CORPUS.md](bench/CORPUS.md): the public test and benchmark texts.
