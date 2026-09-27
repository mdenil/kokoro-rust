# kokoro-rust

Efficient batch text to speech on a local GPU. kokoro-rust runs the
[Kokoro-82M](https://huggingface.co/hexgrad/Kokoro-82M) text-to-speech model natively in Rust and
CUDA: give it a text file with one sentence or paragraph per line, and it reads the whole file
into one WAV.

## Install

```
curl -fsSL https://raw.githubusercontent.com/mdenil/kokoro-rust/main/install.sh | bash
```

This puts the `kokoro` command in `~/.local/bin` and downloads the model. It requires Linux and a
compatible NVIDIA GPU with CUDA installed ([hardware requirements](docs/USAGE.md#requirements)).

## Use

```
kokoro synth book.txt
```

This writes `book.wav` in the current directory.

## More

- [Usage](docs/USAGE.md): options, output files, resuming, requirements and limitations
- [Installing](docs/INSTALL.md): what the installer does, and how to uninstall
- [Building from source](docs/BUILD.md)
- [Validation](docs/VALIDATION.md): how it was checked against the original Python implementation
- [Third-party licenses](docs/THIRD_PARTY.md)
