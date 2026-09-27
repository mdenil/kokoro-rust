# Building from source

Besides the [run-time requirements](USAGE.md#requirements), you need the CUDA Toolkit's `nvcc`
(12.9 tested), a Rust toolchain (1.98 tested), and for the one-time data setup `bash`, `curl`,
`unzip`, `sha256sum` and Python 3.12 with the `venv` module.

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
- `prepare_spacy_assets.sh` creates a Python virtual environment with exact package versions from
  PyPI and the `en_core_web_sm` 3.8.0 model from GitHub, and exports the tokenizer and tagger data.
- Checked against pinned SHA-256 hashes: the model files, the misaki and `en_core_web_sm` wheels,
  the extracted lexicons and the six exported spaCy files. The Python packages of the export
  environment are pinned to exact versions only, not hashes; the exported files are what is
  verified. Both scripts can be re-run; files that already verify are kept.
- eSpeak NG is not downloaded; it comes from your system.

Then point the command at that data:
```
target/release/kokoro synth --model-dir ~/kokoro-data/model --frontend-dir ~/kokoro-data/frontend book.txt
```
This writes `book.wav` in the current directory; see [USAGE.md](USAGE.md) for the options.

Release packages are built with `scripts/package_release.sh` ([INSTALL.md](INSTALL.md#release-packages)).
