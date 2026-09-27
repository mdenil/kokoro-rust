# kokoro-rust

Turn a book into audio on your own NVIDIA GPU. kokoro-rust runs the
[Kokoro-82M](https://huggingface.co/hexgrad/Kokoro-82M) text-to-speech model natively in Rust and
CUDA: give it a text file with one sentence or paragraph per line, and it writes one WAV file per
line.

- Runs locally, with no Python and no cloud service.
- Synthesizes many lines together in each GPU batch.
- Picks up where it left off: running it again skips lines that are already done.
- Never drops a word silently. A line it can't pronounce is reported, so you can fix the text.
- American English, with two voices included.

## Install

```
curl -fsSL https://raw.githubusercontent.com/mdenil/kokoro-rust/main/install.sh | bash
```

This puts the `kokoro` command in `~/.local/bin` and downloads the model. You need Linux on x86_64,
an NVIDIA GPU with compute capability 8.9 (the RTX 40 series, for example) with a driver for CUDA
12.9 and cuBLAS, and eSpeak NG. The installer checks for all of these and offers to install eSpeak
NG on Debian and Ubuntu.

## Use

```
kokoro synth --input book.txt --out-dir out/
```

Each line of `book.txt` becomes `out/book_00001.wav`, `out/book_00002.wav`, and so on.

## More

- [Usage](docs/USAGE.md): options, output files, resuming, requirements and limitations
- [Installing](docs/INSTALL.md): what the installer does, and how to uninstall
- [Building from source](docs/BUILD.md)
- [Validation](docs/VALIDATION.md): how it was checked against the original Python implementation
- [Third-party licenses](docs/THIRD_PARTY.md)
