# kokoro-rust

Turn a book into audio on your own NVIDIA GPU. kokoro-rust runs the
[Kokoro-82M](https://huggingface.co/hexgrad/Kokoro-82M) text-to-speech model natively in Rust and
CUDA: give it a text file with one sentence or paragraph per line, and it writes one WAV file per
line.

- Runs locally, with no Python and no cloud service.
- Picks up where it left off: running it again skips lines that are already done.
- Never drops a word silently. A line it can't pronounce is reported, so you can fix the text.
- American English, with two voices included.

## Install

```
curl -fsSL https://raw.githubusercontent.com/mdenil/kokoro-rust/main/install.sh | bash
```

This puts the `kokoro` command in `~/.local/bin` and downloads the model. It requires Linux and a
compatible NVIDIA GPU with CUDA installed ([hardware requirements](docs/USAGE.md#requirements)).

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
