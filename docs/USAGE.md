# Using kokoro

```
kokoro synth --input book.txt --out-dir out/
```

The installed command finds its model and language data by itself. `--model-dir` and
`--frontend-dir` (or `KOKORO_MODEL_DIR` and `KOKORO_FRONTEND_DIR`) select others.

## Input
UTF-8 text, one utterance per line. Blank lines are errors by default; `--blank-lines skip` records
them without audio instead.

## Output
For `book.txt`:
- `out/book_00001.wav`, `out/book_00002.wav`, …: one file per input line, numbered from 1, 24 kHz
  mono, 16-bit PCM (`--format float32` for 32-bit float).
- `out/book_00001.json`, …: a sidecar per line with the text, phonemes, status, any error, the
  audio hash and the configuration that produced it.
- `out/book.manifest.json`: a summary of the run.

Exit status: 0 when every line was synthesized, 1 when some lines failed (each failure is in its
sidecar and in the manifest), 2 for errors that stop the whole run (missing files, no usable GPU,
missing eSpeak NG).

## Resuming
Running the same command again resumes: a line is skipped only if its WAV still matches the
recorded hash and nothing that affects it (text, voice, speed, model, frontend data, eSpeak NG
installation, …) has changed. `--force` re-synthesizes everything.

## Options
- `--voice am_adam`: voice (default `af_heart`); a `.pt` path or `a,b` (average) also work. Only
  `af_heart` and `am_adam`, the two tested voices, are downloaded.
- `--speed 1.1`: speaking rate.
- `--encode flac` (or `mp3`, `opus`, …): also encode each WAV with `ffmpeg`, which must be
  installed separately.
- `--cuda-device N`: GPU index among the visible devices.
- `--input-format phonemes`: input lines are already phoneme strings (no frontend, no eSpeak NG).
- `--seed N`: seed of the model's noise excitation (default 0).
- Batching and threading (`--batch-phonemes`, `--batch-items`, `--batch-window`,
  `--prep-threads`, `--write-threads`): performance tuning only.

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
  be split, which prints a message. See [design/BATCHING.md](design/BATCHING.md).
- Numerical mode: BF16 tensor-core arithmetic (f32 accumulation) for convolutions and linear layers,
  f32 elsewhere. Output is close to, but not identical with, the reference Python pipeline
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
