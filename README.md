# kokoro-rust

Efficient batch text to speech on a local GPU. kokoro-rust runs the
[Kokoro-82M](https://huggingface.co/hexgrad/Kokoro-82M) text-to-speech model natively in Rust and
CUDA: give it a text file with one sentence or paragraph per line, and it reads the whole file
into one WAV.

## Speed

On an RTX 4090, reading chapter I of *Alice's Adventures in Wonderland* (65 lines, 11 minutes of
audio) into one WAV:

| | Python Kokoro 0.9.4 | kokoro-rust | |
|---|---|---|---|
| Whole command, including startup and model loading | 25.2 s | 2.7 s | 9.4× faster |
| With the model already loaded | 8.0 s | 1.3 s | 6.0× faster |

Medians of 5 runs (model loaded: 6 passes). [How this was measured](docs/VALIDATION.md#performance)

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
- [License](LICENSE): Apache-2.0; [third-party licenses](docs/THIRD_PARTY.md)
