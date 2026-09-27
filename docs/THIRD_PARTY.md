# Third-party components and licenses

This lists what the program and its setup use from others, where it comes from, and the license
stated by its source. It records facts as found; it is not legal advice. The license of this
project's own code has not been decided yet (see README.md).

## Downloaded by the setup scripts (not in this repository)
| component | used for | origin (pinned) | license (as stated by the source) |
|---|---|---|---|
| Kokoro-82M `config.json`, `kokoro-v1_0.pth`, voices `af_heart`, `am_adam` | the model | Hugging Face `hexgrad/Kokoro-82M`, revision `f3ff3571` | Apache-2.0 (model card). The model card names yl4579/StyleTTS2-LJSpeech as the base model and cites the StyleTTS 2 and ISTFTNet papers; the StyleTTS2 code repository is MIT-licensed. |
| misaki 0.9.4 lexicons `us_gold.json`, `us_silver.json` | pronunciation dictionary | the `misaki` 0.9.4 wheel on PyPI | Apache-2.0 (package metadata). The upstream repository does not document where the lexicon entries come from. |
| spaCy en_core_web_sm 3.8.0 (exported tokenizer rules, lookups, tagger weights) | tokenizer and part-of-speech tagger data | `explosion/spacy-models` release on GitHub | MIT (model metadata). The model's metadata lists its training sources: OntoNotes 5 ("commercial (licensed by Explosion)"), WordNet 3.0 (WordNet 3.0 License), and a ClearNLP citation (no code included). |
| spaCy 3.8.14 base norm exceptions (`base_norms.json`) | token normalization data | the `spacy` 3.8.14 package on PyPI | MIT |
| Python packages in the spaCy export environment (`scripts/spacy_export_requirements.txt`) | the one-time export only; not used by the program | PyPI | their own licenses; not part of the program or its output |

## Installed separately by the user
| component | used for | how | license |
|---|---|---|---|
| eSpeak NG (`libespeak-ng.so.1` + `espeak-ng-data`) | pronunciation of words missing from the lexicon | the system package (Debian/Ubuntu: `libespeak-ng1`), loaded at run time with `dlopen`; not included in this repository, not compiled in, not run as a subprocess | GPL-3.0-or-later |
| NVIDIA driver (`libcuda`) and CUDA cuBLAS | GPU execution | the system installation, loaded at run time; `nvcc` from the CUDA toolkit at build time | NVIDIA license terms |
| ffmpeg (optional, `--encode` only) | encoding WAV to other formats | the executable found on PATH, run as a separate process | its own license (LGPL/GPL depending on the build) |

## Compiled into the program
- Rust crates from crates.io, pinned in Cargo.lock: 66 crates (all platforms). Their licenses are
  MIT, Apache-2.0, ISC, Zlib, Unlicense and Unicode-3.0, alone or as alternatives
  (`cargo metadata` shows each crate's license expression). No GPL or MPL crate is included.
- Number words: `src/frontend/num2words.rs` re-implements, for the cases misaki uses, the behaviour
  of the Python package num2words 0.5.14 (LGPL, per its package metadata). It was checked against num2words' output; the
  file states that no num2words code was copied.
- The model and frontend algorithms were implemented from the reference Python packages kokoro
  0.9.4 and misaki 0.9.4 (both Apache-2.0) and transformers 5.12.1 (Apache-2.0, ALBERT encoder).

## In this repository
- `bench/corpus_alice_ch1.txt` and the other *Alice* derivatives: Lewis Carroll, *Alice's Adventures
  in Wonderland*, from Project Gutenberg eBook #11, public domain in the USA (bench/CORPUS.md).
- The per-line audio hash tables under `tests/pinned/` were produced with this program and the
  components above. The reference fixtures the tests read live in the test data root, outside
  this repository.
