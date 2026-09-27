# Using kokoro

```
kokoro synth book.txt
```

This reads `book.txt` and writes `book.wav` in the current directory. Each line of the text file is
one sentence or paragraph, and the WAV contains them in order, one after another, with nothing
added in between. The audio is 24 kHz mono, 16-bit.

`book.txt` can be anywhere; the WAV goes to the directory you run the command in, or to another one
with `--out-dir`. Use `-` as the file name to read the text from standard input (the output is
then `stdin.wav`).

The installed command finds its model and language data by itself. A build from source needs
`--model-dir` and `--frontend-dir`, or `KOKORO_MODEL_DIR` and `KOKORO_FRONTEND_DIR` ([BUILD.md](BUILD.md)).

## The text file
Ordinary written English in UTF-8, one sentence or paragraph per line. A blank line counts as a
mistake by default, so the file isn't written until it's removed; `--blank-lines skip` ignores
blank lines instead.

kokoro never reads a line with a word left out. If a line contains something it can't pronounce,
for example "-12" or a word in another script, it reports that line and writes no WAV. Rewrite the
word ("minus 12") and run the command again.

## Running it again
Running the same command again picks up where it left off: lines that are already done and haven't
changed are reused, and only the rest are synthesized. After you edit a few lines, only those are
redone. `--force` starts over.

This relies on a working directory, `.kokoro/`, next to the output. It holds each line's audio and
how it was made. You can delete it once you have the WAV; the next run then starts from scratch.

## One WAV per line
`--per-line` writes a separate file for each line instead of `book.wav`: `book_00001.wav`,
`book_00002.wav` and so on, numbered by line from 1. Useful when another tool assembles the audio,
or when you want to check or replace single lines.

## JSON details
By default kokoro writes only audio. `--diagnostics` adds JSON files that describe the run:
- with the default output, `book.manifest.json`: the settings, each line's status, text,
  phonemes and hashes, and the WAV's hash and length;
- with `--per-line`, the same manifest plus a JSON file next to each WAV (`book_00001.json`, …).

Adding `--diagnostics` to a finished run writes the JSON without redoing any audio.

## When something fails
Errors always go to the terminal, and the exit status says what happened:
- 0: everything was written;
- 1: some lines failed (they are listed), so `book.wav` was not written;
- 2: the run couldn't start, for example because a file is missing, there's no usable GPU, or
  eSpeak NG isn't installed.

If a `book.wav` from an earlier run is there and the new run fails, the old file stays, and kokoro
says that it doesn't reflect the current text.

## Voice, speed and file format
- `--voice`: `af_heart` (American English, female) by default; `am_adam` (male) is also installed.
  A path to another Kokoro voice file (`.pt`) works too, and `af_heart,am_adam` averages two
  voices. Only the two installed voices are tested.
- `--speed`: 1.0 by default; 1.2 speaks 20% faster, 0.8 slower.
- `--format`: 16-bit samples (`pcm16`) by default; `float32` writes 32-bit float samples.
- `--seed`: 0 by default. The model uses a little random noise as part of the voice; another seed
  gives a slightly different reading.
- `--encode flac` (or `mp3`, `opus`, …): also converts the result with `ffmpeg`, which you need to
  install yourself. That gives `book.flac`, or with `--per-line` one file per line. Off by default;
  plain WAV output doesn't need ffmpeg.

## Text or phonemes
By default kokoro reads written English and works out the pronunciation itself, from its
dictionary and, for words that aren't in it, eSpeak NG. If you want to do that step yourself,
`--input-format phonemes` takes phonemes instead: pronunciation symbols in the form Kokoro uses
(what its own text processing, misaki, produces), up to 510 symbols per line. kokoro then skips its
text processing and doesn't need eSpeak NG.

## GPU and tuning
- `--cuda-device`: which GPU to use, counting those visible to the process; 0 by default.
- `--batch-phonemes`, `--batch-items`, `--batch-window`, `--batch-first-window`, `--prep-threads`
  and `--write-threads` control how lines are grouped for the GPU and how many threads prepare the
  text and write files. The defaults were tuned on an RTX 4090.
- `--fsync` flushes every file to disk before replacing the previous one. It's off by default
  because a rerun checks the saved audio anyway.

`kokoro synth --help` lists everything.

## Requirements
Tested on Ubuntu 24.04 (x86_64) with an RTX 4090, NVIDIA driver 580 and CUDA 12.9. Other GPUs,
distributions and versions are untested ([PORTABILITY.md](PORTABILITY.md)).
- Linux x86_64. Release binaries need glibc 2.35 or newer (e.g. Ubuntu 22.04 or later).
- An NVIDIA GPU with compute capability 8.9 (Ada, e.g. RTX 40-series). Older GPUs are refused at
  startup; newer ones may work but are unverified.
- An NVIDIA driver that supports CUDA 12.9, and cuBLAS from CUDA 12 (`libcublas.so.12`, e.g.
  NVIDIA's `libcublas-12-9` package or the CUDA Toolkit). The rest of the CUDA Toolkit is not
  needed to run kokoro.
- eSpeak NG, from your distribution (Debian/Ubuntu: `sudo apt install libespeak-ng1`).
- Optional: `ffmpeg`, only for `--encode`.

## Limitations
- American English only, and only on an NVIDIA GPU (there is no CPU mode).
- Words that aren't in the dictionary are pronounced by eSpeak NG, so their pronunciation can
  change with the installed eSpeak NG version.
- A line whose phonemes can't be split into pieces of at most 510 symbols is refused rather than
  cut off.
- The audio depends very slightly on which lines are synthesized together. Repeating a run gives
  identical files, unless GPU memory runs short and kokoro has to split a batch (it says so). Lines
  redone after an edit can sound very slightly different from a full run.
- A single WAV file can hold at most 4 GiB: about 24 hours of 16-bit audio, 12 hours of `float32`.
  Longer input is refused; split it into several files, or use `--per-line`.
- The model runs in reduced precision (BF16 matrix arithmetic with f32 accumulation, f32
  elsewhere), so the audio is close to, but not identical with, the original Python implementation
  ([VALIDATION.md](VALIDATION.md)).

## Technical documentation
- [PORTABILITY.md](PORTABILITY.md): configuration reference, tested scope, and how to run the tests.
- [DEPENDENCIES.md](DEPENDENCIES.md): what the program loads at run time.
- [VALIDATION.md](VALIDATION.md): how the port was checked against the reference, and measured
  performance.
- [design/BATCHING.md](design/BATCHING.md): batched inference.
- [frontend/](frontend/): the text frontend specification and coverage.
- [truth-pack/](truth-pack/): the pinned upstream sources and hashes.
- [../bench/CORPUS.md](../bench/CORPUS.md): the public test and benchmark texts.
